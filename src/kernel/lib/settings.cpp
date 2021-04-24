/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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
    // Throw exception if not.
    void Dims::check_dim_type(const std::string& dim,
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

    // Debug & trace.
    omp_lock_t KernelEnv::_debug_lock;
    bool KernelEnv::_debug_lock_init_done = false;
    yask_output_ptr KernelEnv::_debug;
    bool KernelEnv::_trace = false;

    // OMP offload devices.
    #ifdef USE_OFFLOAD
    bool KernelEnv::_use_offload = true;
    int KernelEnv::_omp_hostn = 0;
    int KernelEnv::_omp_devn = 0;
    #else
    bool KernelEnv::_use_offload = false;
    #endif

    // Debug APIs.
    yask_output_ptr yk_env::get_debug_output() {
        if (!KernelEnv::_debug.get()) {
            yask_output_factory ofac;
            auto so = ofac.new_stdout_output();
            set_debug_output(so);
        }
        assert(KernelEnv::_debug.get());
        return KernelEnv::_debug;
    }
    void yk_env::set_debug_output(yask_output_ptr debug) {
        KernelEnv::_debug = debug;
    }
    void yk_env::set_trace_enabled(bool enable) {
        KernelEnv::_trace = enable;
    }
    bool yk_env::is_trace_enabled() {
        return KernelEnv::_trace;
    }
    
    // Apply a function to each neighbor rank.
    // Does NOT visit self.
    void MPIInfo::visit_neighbors(std::function<void
                                 (const IdxTuple& neigh_offsets, // NeighborOffset vals.
                                  int neigh_rank, // MPI rank.
                                  int neigh_index)> visitor) {

        neighborhood_sizes.visit_all_points
            ([&](const IdxTuple& neigh_offsets, size_t i) {
                 int neigh_rank = my_neighbors.at(i);
                 assert(i == get_neighbor_index(neigh_offsets));

                 if (i != my_neighbor_index)
                     visitor(neigh_offsets, neigh_rank, i);
                 return true; // from lambda;
             });
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
    void MPIData::visit_neighbors(std::function<void
                                 (const IdxTuple& neigh_offsets, // NeighborOffset.
                                  int neigh_rank,
                                  int neigh_index,
                                  MPIBufs& bufs)> visitor) {

        _mpi_info->visit_neighbors
            ([&](const IdxTuple& neigh_offsets, int neigh_rank, int i) {

                 if (neigh_rank != MPI_PROC_NULL)
                     visitor(neigh_offsets, neigh_rank, i, bufs[i]);
             });
    }

    // Access a buffer by direction and neighbor offsets.
    MPIBuf& MPIData::get_buf(MPIBufs::BufDir bd, const IdxTuple& offsets) {
        assert(int(bd) < int(MPIBufs::n_buf_dirs));
        auto i = _mpi_info->get_neighbor_index(offsets); // 1D index.
        assert(i < _mpi_info->neighborhood_size);
        return bufs[i].bufs[bd];
    }

    // Settings ctor.
    KernelSettings::KernelSettings(DimsPtr dims, KernelEnvPtr env) :
        _dims(dims), max_threads(env->max_threads) {
        auto& step_dim = dims->_step_dim;

        // Target-dependent defaults.
        int def_blk_size = 32;  // TODO: calculate based on actual cache size and stencil.
        num_block_threads = 1;
        if (string(YASK_TARGET) == "knl") {
            def_blk_size = 64;   // larger L2.
            num_block_threads = 8; // 4 threads per core * 2 cores per tile.
        }

        // Use both step and domain dims for all size tuples.
        _global_sizes = dims->_stencil_dims;
        _global_sizes.set_vals_same(0); // 0 => calc from rank.

        _rank_sizes = dims->_stencil_dims;
        _rank_sizes.set_vals_same(0); // 0 => calc from global.
        _rank_tile_sizes = dims->_stencil_dims;
        _rank_tile_sizes.set_vals_same(0); // 0 => rank size.

        _region_sizes = dims->_stencil_dims;
        _region_sizes.set_vals_same(0);   // 0 => rank size.
        _region_tile_sizes = dims->_stencil_dims;
        _region_tile_sizes.set_vals_same(0); // 0 => region size.
        
        _block_sizes = dims->_stencil_dims;
        _block_sizes.set_vals_same(def_blk_size);
        _block_sizes.set_val(step_dim, 0); // 0 => default.
        _block_tile_sizes = dims->_stencil_dims;
        _block_tile_sizes.set_vals_same(0); // 0 => block size.

        _mini_block_sizes = dims->_stencil_dims;
        _mini_block_sizes.set_vals_same(0);            // 0 => calc from block.
        _mini_block_tile_sizes = dims->_stencil_dims;
        _mini_block_tile_sizes.set_vals_same(0); // 0 => mini-block size.

        _sub_block_sizes = dims->_stencil_dims;
        _sub_block_sizes.set_vals_same(0);            // 0 => calc from mini-block.
        _sub_block_tile_sizes = dims->_stencil_dims;
        _sub_block_tile_sizes.set_vals_same(0); // 0 => sub-block size.

        _min_pad_sizes = dims->_stencil_dims;
        _min_pad_sizes.set_vals_same(0);

        _extra_pad_sizes = dims->_stencil_dims;
        _extra_pad_sizes.set_vals_same(0);

        // Use only domain dims for MPI tuples.
        _num_ranks = dims->_domain_dims;
        _num_ranks.set_vals_same(0); // 0 => set using heuristic.

        _rank_indices = dims->_domain_dims;
        _rank_indices.set_vals_same(0);
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
            auto& dname = dim._get_name();
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
        _add_domain_option(parser, "d", "[Deprecated] Alias for local-domain size", _rank_sizes);
        _add_domain_option(parser, "r", "Region size", _region_sizes, true);
        _add_domain_option(parser, "b", "Block size", _block_sizes, true);
        _add_domain_option(parser, "mb", "Mini-block size", _mini_block_sizes);
        _add_domain_option(parser, "sb", "Sub-block size", _sub_block_sizes);
#ifdef USE_TILING
        _add_domain_option(parser, "l_tile", "[Advanced] Local-domain-tile size", _rank_tile_sizes);
        _add_domain_option(parser, "r_tile", "[Advanced] Region-tile size", _region_tile_sizes);
        _add_domain_option(parser, "b_tile", "[Advanced] Block-tile size", _block_tile_sizes);
        _add_domain_option(parser, "mb_tile", "[Advanced] Mini-block-tile size", _mini_block_tile_sizes);
        _add_domain_option(parser, "sb_tile", "[Advanced] Sub-block-tile size", _sub_block_tile_sizes);
#endif
        _add_domain_option(parser, "mp", "[Advanced] Minimum var-padding size (including halo)", _min_pad_sizes);
        _add_domain_option(parser, "ep", "[Advanced] Extra var-padding size (beyond halo)", _extra_pad_sizes);
        parser.add_option(new CommandLineParser::BoolOption
                          ("allow_addl_padding",
                           "[Advanced] Allow automatic extension of padding beyond what is needed for"
                           " vector alignment for additional performance reasons",
                           _allow_addl_pad));
#ifdef USE_MPI
        _add_domain_option(parser, "nr", "Num ranks", _num_ranks);
        _add_domain_option(parser, "ri", "This rank's logical index (0-based)", _rank_indices);
        parser.add_option(new CommandLineParser::BoolOption
                          ("overlap_comms",
                           "Overlap MPI communication with calculation of interior elements whenever possible.",
                           overlap_comms));
        parser.add_option(new CommandLineParser::IdxOption
                          ("min_exterior",
                           "[Advanced] Minimum width of exterior section to"
                           " compute before starting MPI communication. "
                           "Applicable only when overlap_comms is enabled.",
                           _min_exterior));
        parser.add_option(new CommandLineParser::BoolOption
                          ("use_shm",
                           "Directly use shared memory for halo-exchange buffers "
                           "between ranks on the same node when possible. "
                           "Otherwise, use the same non-blocking MPI send and receive calls "
                           "that are used between nodes.",
                           use_shm));
#endif
        parser.add_option(new CommandLineParser::BoolOption
                          ("force_scalar",
                           "[Advanced] Evaluate every var point with scalar stencil operations "
                           "and exchange halos using only scalar packing and unpacking (for debug).",
                           force_scalar));
        parser.add_option(new CommandLineParser::IntOption
                          ("max_threads",
                           "Maximum OpenMP threads to use. "
                           "If set to zero (0), the default value from the OpenMP library is used.",
                           max_threads));
        parser.add_option(new CommandLineParser::IntOption
                          ("thread_divisor",
                           "Divide the maximum number of OpenMP threads by the specified value, "
                           "discarding any remainder. "
                           "The maximum number of OpenMP threads is determined by the -max_threads "
                           "option or the default value from the OpenMP library. ",
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
                           "[Advanced] Divide mini-blocks into sub-blocks of slabs along the first valid dimension "
                           "(usually the outer-domain dimension), ignoring other sub-block sizes. "
                           "Assign each slab to a block thread based on its global index in that dimension. "
                           "This setting may increase cache locality when using multiple "
                           "block-threads, especially when scratch vars are used and/or "
                           "when temporal blocking is active. "
                           "This option is ignored if there are fewer than two block threads.",
                           bind_block_threads));
#ifdef USE_NUMA
        parser.add_option(new CommandLineParser::IntOption
                          ("numa_pref",
                           "[Advanced] Preferred NUMA node on which to allocate data for "
                           "vars and MPI buffers. Alternatively, use special values " +
                           to_string(yask_numa_local) + " for explicit local-node allocation, " +
                           to_string(yask_numa_interleave) + " for interleaving pages across all nodes, or " +
                           to_string(yask_numa_none) + " for no NUMA policy.",
                           _numa_pref));
#endif
#ifdef USE_PMEM
        parser.add_option(new CommandLineParser::IntOption
                          ("pmem_threshold",
                           "[Advanced] First allocate up to this many GiB for vars using system memory, "
                           "then allocate memory for remaining vars from a PMEM (persistent memory) device "
                           "named '/mnt/pmemX', where 'X' corresponds to the NUMA node of the YASK process.",
                           _numa_pref_max));
#endif
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune",
                           "Adjust specified block and tile sizes *during* normal operation "
                           "to tune for performance, a.k.a., 'online' or 'in-situ' tuning. "
                           "Successive steps will use different sizes until an optimal size is found. "
                           "Will likely cause varying performance between steps, "
                           "so this is not recommended for benchmarking. "
                           "However, this can be a useful tool for deployment of YASK stencils "
                           "onto unknown and/or varied systems where 'offline' tuning is not practical.",
                           _do_auto_tune));
        parser.add_option(new CommandLineParser::DoubleOption
                          ("auto_tune_trial_secs",
                           "[Advanced] Seconds to run new trial during auto-tuning "
                           "for new trial to be considered better than the existing best.",
                           _tuner_trial_secs));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_each_stage",
                           "[Advanced] Apply the auto-tuner separately to each stage. "
                           "Will only be used if stages are applied in separate "
                           "passes across the entire var, "
                           "i.e., when no temporal tiling is used.",
                           _allow_stage_tuners));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_blocks",
                           "[Advanced] Apply the auto-tuner to block sizes. "
                           "This is the most common tuning for CPU kernels. "
                           "Auto-tuning blocks will automatically set all lower-level "
                           "sizes to their defaults after each change to the block sizes.",
                           _tune_blks));
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_mini_blocks",
                           "[Advanced] Apply the auto-tuner to mini-block sizes. "
                           "Often useful when using temporal block tiling."
                           "Auto-tuning mini-blocks will automatically set all lower-level "
                           "sizes to their defaults after each change to the mini-block sizes.",
                           _tune_mini_blks));
        #ifdef USE_TILING
        parser.add_option(new CommandLineParser::BoolOption
                          ("auto_tune_sub_block_tiles",
                           "[Advanced] Apply the auto-tuner to sub-block-tile sizes. "
                           "Often useful for tuning GPU kernels. "
                           "Auto-tuning sub-block-tiles will automatically set all lower-level "
                           "sizes to their defaults after each change to the sub-block-tile sizes.",
                           _tune_sub_blk_tiles));
        #endif
    }

    // Print usage message.
    void KernelSettings::print_usage(ostream& os,
                                     CommandLineParser& app_parser,
                                     const string& pgm_name,
                                     const string& app_notes,
                                     const vector<string>& app_examples)
    {
        os << "Usage: " << pgm_name << " [options]\n"
            "Options:\n";
        app_parser.print_help(os);
        CommandLineParser soln_parser;
        add_options(soln_parser);
        soln_parser.print_help(os);
        os <<
            "\nTerms for the various work-sizes from largest to smallest:\n"
            " The 'global-domain' or 'overall-problem' is the work done across all MPI ranks.\n"
            "  The global-domain is composed of one or more local-domains.\n"
            #ifndef USE_MPI
            "  This binary has NOT been compiled with MPI support,\n"
            "   so the global-domain is equivalent to the single local-domain.\n"
            #endif
            " A 'local-domain' or 'rank-domain' is the work done in one MPI rank.\n"
            "  The purpose of local-domains is to control the amount of work done in one\n"
            "   entire MPI rank.\n"
            "  Ranks are evaluated in parallel in separate MPI processes.\n"
            "  Each local-domain is composed of one or more regions.\n"
            " A 'region' is a sub-division of work within a local-domain.\n"
            "  The purpose of regions is to control the amount of work done across an\n"
            "   entire MPI rank while sharing a large cache.\n"
            "  If using temporal wave-front rank tiling (see region-size guidelines),\n"
            "   then this is the work done in each wave-front rank tile;\n"
            "   else, there is typically only one region the size of the local-domain.\n"
            "  Regions are evaluated sequentially within ranks.\n"
            "  Each region is composed of one or more blocks.\n"
            " A 'block' is a sub-division of work within a region.\n"
            "  The purpose of blocking is to provide control over the amount of\n"
            "   work done by each concurrent OpenMP thread.\n"
            "  If the number of threads is greater than one (typical),\n"
            "   then this is the unit of work for one OpenMP thread,\n"
            "   and blocks are evaluated concurrently within each region;\n"
            "   else, blocks are evaluated sequentially within each region.\n"
            "  This is the most commonly-tuned work-size for many stencils, especially\n"
            "   when not using any sort of temporal tiling.\n"
            "  Each block is composed of one or more mini-blocks.\n"
            " A 'mini-block' is a sub-division of work within a block.\n"
            "  The purpose of mini-blocking is to allow distinction between the amount\n"
            "   of work done by a thread (via blocking) and the amount of work done for\n"
            "   cache locality (via mini-blocking).\n"
            "  If using temporal wave-front block tiling (see mini-block-size guidelines),\n"
            "   then this is the work done for each wave-front block tile,\n"
            "   and the number temporal steps in the mini-block is always equal\n"
            "   to the number temporal steps a block;\n"
            "   else, there is typically only one mini-block the size of a block.\n"
            "  If mini-block sizes are not specified, a mini-block is the same size\n"
            "   as a block, and the amount of work done by a thread and the amount of\n"
            "   work done for cache locality is the same.\n"
            "  Mini-blocks are evaluated sequentially within blocks.\n"
            "  Each mini-block is composed of one or more sub-blocks.\n"
            " A 'sub-block' is a sub-division of work within a mini-block.\n"
            "  The purpose of sub-blocking is to allow multiple OpenMP threads\n"
            "   to cooperatively work on a mini-block, sharing cached values--\n"
            "   this is particularly useful when using hyper-threads on a CPU.\n"
            "  If the number of block-threads is greater than one,\n"
            "   then this is the unit of work for one nested OpenMP thread,\n"
            "   and sub-blocks are evaluated concurrently within each mini-block;\n"
            "   else, sub-blocks are evaluated sequentially within each mini-block.\n"
            #ifdef USE_OFFLOAD
            "  When offloading computation to a device, this is the unit of work\n"
            "   done in each offloaded kernel invocation.\n"
            #endif
            "  There is no temporal tiling at the sub-block level.\n"
            "  Each sub-block is composed of one or more clusters.\n"
            " A 'cluster' is the work done in each inner-most loop iteration.\n"
            "  The purpose of clustering is to allow more than one vector of\n"
            "   work to be done in each loop iteration, useful for very simple stencils.\n"
            "  Each cluster is composed of one or more vectors.\n"
            " A 'vector' is typically the work done by a SIMD instruction.\n"
            "  A 'folded vector' contains points in more than one dimension.\n"
            "  The size of a vector is typically that of a SIMD register.\n"
            "  Each vector is composed of one or more points.\n"
            " A 'point' is a single floating-point (FP) element in a grid.\n"
            "  This binary uses " << REAL_BYTES << "-byte FP elements.\n"
            "\n"
            "Guidelines for setting work-sizes and their defaults:\n"
            " The global-domain sizes specify the work done across all MPI ranks.\n"
            "  A global-domain size of 0 in a given domain dimension =>\n"
            "   global-domain size is the sum of local-domain sizes in that dimension.\n"
            " The local-domain sizes specify the work done on each MPI rank.\n"
            "  A local-domain size of 0 in a given domain dimension =>\n"
            "   local-domain size is determined by the global-domain size in that dimension.\n"
            "  This and the number of vars affect the amount of memory used per rank.\n"
            "  Either the global-domain size or the local-domain size must be specified.\n"
            " The region sizes are used to configure temporal wave-front rank tiling.\n"
            "  Temporal wave-front rank tiling may increase locality in large shared caches\n"
            "   when a local-domain is larger than the capacity of those caches.\n"
            "  A region size >1 in the step dimension (e.g., '-rt') enables wave-front rank tiling.\n"
            "  A region size of 0 in the step dimension => the temporal wave-front\n"
            "   rank tiling will have the same number of steps as the temporal block tiling.\n"
            "  The region size in the step dimension affects how often MPI halo-exchanges occur:\n"
            "   A region size of 0 in the step dimension => exchange after every stage.\n"
            "   A region size >0 in the step dimension => exchange after that many steps.\n"
            " The block sizes specify the work done by each top-level OpenMP thread.\n"
            "  A block size of 0 in a given domain dimension =>\n"
            "   block size is set to region size in that dimension.\n"
            "  A block size of 0 in the step dimension (e.g., '-bt') disables any temporal blocking.\n"
            "  A block size of 1 in the step dimension enables temporal blocking, but only between\n"
            "   stages in the same step.\n"
            "  A block size >1 in the step dimension enables temporal region tiling across multiple steps.\n"
            "  The temporal block size may be automatically reduced if needed based on the\n"
            "   domain block sizes, the stencil halos, and the step size of the regions.\n"
            " The mini-block sizes are used to configure temporal wave-front block tiling.\n"
            "   Temporal wave-front block tiling may increase locality in core-local caches\n"
            "   (e.g., L2) when blocks are larger than that the capacity of those caches.\n"
            "  A mini-block size of 0 in a given domain dimension =>\n"
            "   mini-block size is set to block size in that dimension.\n"
            "  The size of a mini-block in the step dimension is always implicitly\n"
            "   the same as that of a block.\n"
            " The sub-block sizes specify the work done by each nested OpenMP thread.\n"
            "  Multiple sub-blocks may enable more effective sharing of caches\n"
            "   among multiple hyper-threads in a core when there is more than\n"
            "   one block-thread. It can also be used to share data between caches\n"
            "   among multiple cores.\n"
            "  A sub-block size of 0 in a given domain dimension =>\n"
            "   sub-block size is set to mini-block size in that dimension;\n"
            "   when there is more than one block-thread, the first dimension\n"
            "   will instead be set to the vector length to create \"slab\" shapes.\n"
            " The vector and cluster sizes are set at compile-time, so\n"
            "  there are no run-time options to set them.\n"
            #ifdef USE_TILING
            " Set 'tile' sizes to provide finer control over the order of evaluation\n"
            "  within the given area. For example, sub-block-tiles create smaller areas\n"
            "  within sub-blocks; points with the first sub-block-tile will be scheduled\n"
            "  before those the second sub-block-tile, etc. (There is no additional level\n"
            "  of temporal tiling or sychronization added with this tiling.)\n"
            "  A tile size of 0 in a given domain dimension => tile size is set to the size\n"
            "   of its enclosing area in that dimension, i.e., there will only be one tile\n"
            "   in that dimension.\n"
            #endif
            #ifdef USE_MPI
            "\nControlling MPI scaling:\n"
            " To 'strong-scale' a given overall-problem size, use multiple MPI ranks\n"
            "  and keep the global-domain sizes constant.\n"
            " To 'weak-scale' to a larger overall-problem size, use multiple MPI ranks\n"
            "  and keep the local-domain sizes constant.\n"
            #endif
            "\nControlling OpenMP threading:\n"
            " Using '-max_threads 0' =>\n"
            "  max_threads is set to OpenMP's default number of threads.\n"
            " The -thread_divisor option is a convenience to reduce the number of\n"
            "  threads used without having to know the max_threads setting;\n"
            "  e.g., using '-thread_divisor 2' will halve the number of OpenMP threads.\n"
            " For stencil evaluation, threads are allocated using nested OpenMP:\n"
            "  Num CPU threads per mini-block and sub-block = 1.\n"
            "  Num CPU threads per block = block_threads.\n"
            "  Num CPU threads per region = max_threads / thread_divisor / block_threads.\n"
            #ifdef USE_OFFLOAD
            " When using offloaded kernel evaluation, there may be multiple teams\n"
            "  and offload threads used within each sub-block. These may be controlled by\n"
            "  the standard OpenMP environment vars OMP_NUM_TEAMS and OMP_TEAMS_THREAD_LIMIT.\n"
            #endif
           << app_notes;

        // Make example knobs.
        string ex1, ex2;
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = _dims->_domain_dims.get_dim_name(j);
            ex1 += " -g" + dname + " " + to_string(i * 128);
            ex2 += " -nr" + dname + " " + to_string(i + 1);
        }
        os <<
            "\nExamples:\n"
            " " << pgm_name << " -g 768  # global-domain size in all dims.\n"
            " " << pgm_name << ex1 << "  # global-domain size in each dim.\n"
            " " << pgm_name << " -l 2048 -r 512 -rt 10  # local-domain size and temporal rank tiling.\n"
            " " << pgm_name << " -g 512" << ex2 << "  # number of ranks in each dim.\n";
        for (auto ae : app_examples)
            os << " " << pgm_name << " " << ae << endl;
        os << flush;
    }

    // For each one of 'inner_sizes' dim that is zero,
    // make it equal to same dim in 'outer_sizes'.
    // Round up each of 'inner_sizes' dim to be a multiple of same dim in 'mults'.
    // Limit size to 'outer_sizes'.
    // Output info to 'os' using '*_name' and dim names.
    // Does not process 'step_dim'.
    // Return product of number of inner subsets.
    static idx_t find_num_subsets(ostream& os,
                                  IdxTuple& inner_sizes, const string& inner_name,
                                  const IdxTuple& outer_sizes, const string& outer_name,
                                  const IdxTuple& mults, const string& mult_name,
                                  const std::string& step_dim) {

        idx_t prod = 1;
        bool rounded = false;
        bool trimmed = false;
        for (auto& dim : inner_sizes) {
            auto& dname = dim._get_name();
            if (dname == step_dim)
                continue;
            idx_t* dptr = inner_sizes.lookup(dname); // use lookup() to get non-const ptr.

            // Set default to outer size.
            idx_t outer_size = outer_sizes[dname];
            if (*dptr <= 0)
                *dptr = outer_size; // 0 => use full size as default.

            // Round up.
            if (mults.lookup(dname) && mults[dname] > 1) {
                idx_t rsz = ROUND_UP(*dptr, mults[dname]);
                if (rsz != *dptr) {
                    *dptr = rsz;
                    rounded = true;
                }
            }

            // Limit.
            if (*dptr > outer_size) {
                *dptr = outer_size;
                trimmed = true;
            }

            // Calc stats.
            idx_t inner_size = *dptr;
            idx_t ninner = (inner_size <= 0) ? 0 :
                (outer_size + inner_size - 1) / inner_size; // full or partial.
            idx_t rem = (inner_size <= 0) ? 0 :
                outer_size % inner_size;                       // size of remainder.
            idx_t nfull = rem ? (ninner - 1) : ninner; // full only.

            if (outer_size > 0) {
                os << " In '" << dname << "' dim, " <<
                    outer_name << " of size " <<
                    outer_size << " contains " << ninner << " " <<
                    inner_name << "(s)";
                if (rem)
                    os << ": " << nfull << " of full-size " << inner_size <<
                        " plus 1 of remainder-size " << rem;
                else
                    os << " of size " << inner_size;
                os << ".\n";
            }
            prod *= ninner;
        }
        if (rounded)
            os << " The " << inner_name << " sizes have been rounded up to multiples of " <<
                mult_name << " sizes.\n";
        if (trimmed)
            os << " The " << inner_name << " sizes have been limited to " <<
                outer_name << " sizes.\n";
        return prod;
    }

    // Make sure all user-provided settings are valid and finish setting up some
    // other vars before allocating memory.
    // Called from prepare_solution(), during auto-tuning, etc.
    void KernelSettings::adjust_settings(KernelStateBase* ksb) {
        yask_output_ptr op = ksb ? ksb->get_debug_output() : nullop;
        ostream& os = op->get_ostream();

        auto& step_dim = _dims->_step_dim;
        auto& inner_dim = _dims->_inner_dim;
        auto& rt = _region_sizes[step_dim];
        auto& bt = _block_sizes[step_dim];
        auto& mbt = _mini_block_sizes[step_dim];
        auto& cluster_pts = _dims->_cluster_pts;
        int nddims = _dims->_domain_dims.get_num_dims();

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
        auto nr = find_num_subsets(os,
                                   _region_sizes, "region",
                                   _rank_sizes, "local-domain",
                                   cluster_pts, "cluster",
                                   step_dim);
        os << " num-regions-per-local-domain-per-step: " << nr << endl;
        os << " Since the region size in the '" << step_dim <<
            "' dim is " << rt << ", temporal wave-front rank tiling is ";
        if (!rt) os << "NOT ";
        os << "enabled.\n";

        // Determine num blocks.
        // Also fix up block sizes as needed.
        // Default block size (if 0) will be size of region.
        os << "\nBlocks:" << endl;
        auto nb = find_num_subsets(os,
                                   _block_sizes, "block",
                                   _region_sizes, "region",
                                   cluster_pts, "cluster",
                                   step_dim);
        os << " num-blocks-per-region-per-step: " << nb << endl;
        os << " num-blocks-per-local-domain-per-step: " << (nb * nr) << endl;
        os << " Since the block size in the '" << step_dim <<
            "' dim is " << bt << ", temporal concurrent region tiling is ";
        if (!bt) os << "NOT ";
        os << "enabled.\n";

        // Determine num mini-blocks.
        // Also fix up mini-block sizes as needed.
        os << "\nMini-blocks:" << endl;
        auto nmb = find_num_subsets(os,
                                    _mini_block_sizes, "mini-block",
                                    _block_sizes, "block",
                                    cluster_pts, "cluster",
                                    step_dim);
        os << " num-mini-blocks-per-block-per-step: " << nmb << endl;
        os << " num-mini-blocks-per-region-per-step: " << (nmb * nb) << endl;
        os << " num-mini-blocks-per-local-domain-per-step: " << (nmb * nb * nr) << endl;
        os << " Since the mini-block size in the '" << step_dim <<
            "' dim is " << mbt << ", temporal wave-front block tiling is ";
        if (!mbt) os << "NOT ";
        os << "enabled.\n";

        // Adjust defaults for sub-blocks to be slab if we are using more
        // than one block thread.  Otherwise, find_num_subsets() would set
        // default to entire block, and we wouldn't use multiple threads.
        if (num_block_threads > 1 && _sub_block_sizes.sum() == 0) {

            // Default dim is outer one.
            _bind_posn = 1;

            // Look for best dim to split and bind threads to
            // if binding is enabled.
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.get_dim_name(j);
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
        auto nsb = find_num_subsets(os,
                                    _sub_block_sizes, "sub-block",
                                    _mini_block_sizes, "mini-block",
                                    cluster_pts, "cluster",
                                    step_dim);
        os << " num-sub-blocks-per-mini-block-per-step: " << nsb << endl;
        os << " num-sub-blocks-per-block-per-step: " << (nsb * nmb) << endl;
        os << " num-sub-blocks-per-region-per-step: " << (nsb * nmb * nb) << endl;
        os << " num-sub-blocks-per-rank-per-step: " << (nsb * nmb * nb * nr) << endl;
        os << " Temporal mini-block tiling is never enabled.\n";

        // Determine binding dimension. Do this again if it was done above
        // by default because it may have changed during adjustment.
        if (bind_block_threads && num_block_threads > 1) {
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.get_dim_name(j);
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
                _dims->_stencil_dims.get_dim_name(_bind_posn) << "' dimension may be used at run-time\n"
                "  because block-thread binding is enabled on " << num_block_threads << " block threads.\n";
        }

#ifdef USE_TILING
        // Now, we adjust tiles. These are done after all the above sizes
        // because tile sizes are more like 'guidelines' and don't have
        // their own loops.

        // Show num rank-tiles.
        // TODO: only print this if rank-tiling is enabled.
        os << "\nLocal-domain tiles:\n";
        auto nlg = find_num_subsets(os,
                                    _rank_tile_sizes, "local-domain-tile",
                                    _rank_sizes, "local-domain",
                                    _region_sizes, "region",
                                    step_dim);
        os << " num-local-domain-tiles-per-local-domain-per-step: " << nlg << endl;

        // Show num region-tiles.
        // TODO: only print this if region-tiling is enabled.
        os << "\nRegion tiles:\n";
        auto nrg = find_num_subsets(os,
                                    _region_tile_sizes, "region-tile",
                                    _region_sizes, "region",
                                    _block_sizes, "block",
                                    step_dim);
        os << " num-region-tiles-per-region-per-step: " << nlg << endl;

        // Show num block-tiles.
        // TODO: only print this if block-tiling is enabled.
        os << "\nBlock tiles:\n";
        auto nbg = find_num_subsets(os,
                                    _block_tile_sizes, "block-tile",
                                    _block_sizes, "block",
                                    _mini_block_sizes, "mini-block",
                                    step_dim);
        os << " num-block-tiles-per-block-per-step: " << nbg << endl;

        // Show num mini-block-tiles.
        // TODO: only print this if mini-block-tiling is enabled.
        os << "\nMini-block tiles:\n";
        auto nmbt = find_num_subsets(os,
                                     _mini_block_tile_sizes, "mini-block-tile",
                                     _mini_block_sizes, "mini-block",
                                     _sub_block_sizes, "sub-block",
                                     step_dim);
        os << " num-mini-block-tiles-per-mini-block-per-step: " << nmbt << endl;

        // Show num sub-block-tiles.
        // TODO: only print this if sub-block-tiling is enabled.
        os << "\nSub-block tiles:\n";
        auto nsbt = find_num_subsets(os,
                                     _sub_block_tile_sizes, "sub-block-tile",
                                     _sub_block_sizes, "sub-block",
                                     cluster_pts, "cluster",
                                     step_dim);
        os << " num-sub-block-tiles-per-sub-block-per-step: " << nsbt << endl;
#endif
        os << endl;
    }

    // Ctor.
    KernelStateBase::KernelStateBase(KernelEnvPtr& kenv,
                                     KernelSettingsPtr& ksettings)
    {
        // Create state. All other objects that need to share
        // this state should use a shared ptr to it.
        _state = make_shared<KernelState>();

        // Share passed ptrs.
        host_assert(kenv);
        _state->_env = kenv;
        host_assert(ksettings);
        _state->_opts = ksettings;
        host_assert(ksettings->_dims);
        _state->_dims = ksettings->_dims;

       // Create MPI Info object.
        _state->_mpi_info = make_shared<MPIInfo>(ksettings->_dims);

        // Set vars after above inits.
        STATE_VARS(this);

        // Find index posns in stencil dims.
        DOMAIN_VAR_LOOP(i, j) {
            auto& dname = stencil_dims.get_dim_name(i);
            if (state->_outer_posn < 0)
                state->_outer_posn = i;
            if (dname == dims->_inner_dim)
                state->_inner_posn = i;
        }
        host_assert(outer_posn == state->_outer_posn);
    }

    // Set number of threads w/o using thread-divisor.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_max_threads() {
        STATE_VARS(this);

        // Get max number of threads.
        int mt = max(opts->max_threads, 1);

        // Set num threads to use for inner and outer loops.
        yask_num_threads[0] = mt;
        yask_num_threads[1] = 0;

        // Reset number of OMP threads to max allowed.
        omp_set_num_threads(mt);
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
    // Enable nested OMP.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_region_threads() {
        int rt=0, bt=0;
        int at = get_num_comp_threads(rt, bt);

        // Must call before entering top parallel region.
        int ol = omp_get_level();
        assert(ol == 0);

        // Enable nested OMP.
        omp_set_nested(1);
        omp_set_max_active_levels(yask_max_levels + 1); // Add 1 for offload.
         
        // Set num threads to use for inner and outer loops.
        yask_num_threads[0] = rt;
        yask_num_threads[1] = bt;

        // Set num threads for a region.
        omp_set_num_threads(rt);
        return rt;
    }

    
    // Set number of threads for a block.
    // Must be called from within a top-level OMP parallel region.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_block_threads() {
        int rt=0, bt=0;
        int at = get_num_comp_threads(rt, bt);

        // Must call within top parallel region.
        #ifdef _OPENMP
        int ol = omp_get_level();
        assert(ol == 1);
        int mal = omp_get_max_active_levels();
        assert (mal >= 2);
        #endif

        omp_set_num_threads(bt);
        return bt;
    }


    // ContextLinker ctor.
    ContextLinker::ContextLinker(StencilContext* context) :
        KernelStateBase(context->get_state()),
        _context(context) {
        assert(context);
    }

} // namespace yask.
