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
    void yk_env::disable_debug_output() {
        yask_output_factory yof;
        KernelEnv::_debug = yof.new_null_output();
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
            ([&](const IdxTuple& neigh_offsets, idx_t i) {
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

    // Settings static vars.
    const string KernelSettings::_mega_block_str = "Mb";
    const string KernelSettings::_block_str = "b";
    const string KernelSettings::_micro_block_str = "mb";
    const string KernelSettings::_nano_block_str = "nb";
    const string KernelSettings::_pico_block_str = "pb";
    
    // Settings ctor.
    KernelSettings::KernelSettings(DimsPtr dims, KernelEnvPtr env) :
        _dims(dims), max_threads(env->max_threads) {
        auto& step_dim = dims->_step_dim;

        // Target-dependent defaults.
        def_blk_size = 32;  // TODO: calculate based on actual cache size and stencil.
        num_inner_threads = 1;
        if (string(YASK_TARGET) == "knl") {
            def_blk_size = 64;   // larger L2.
            num_inner_threads = 8; // 4 threads per core * 2 cores per tile.
        }

        // Use both step and domain dims for all size tuples.
        _global_sizes = dims->_stencil_dims;
        _global_sizes.set_vals_same(0); // 0 => calc from rank.

        _rank_sizes = dims->_stencil_dims;
        _rank_sizes.set_vals_same(0); // 0 => calc from global.
        _rank_tile_sizes = dims->_stencil_dims;
        _rank_tile_sizes.set_vals_same(0); // 0 => rank size.

        _mega_block_sizes = dims->_stencil_dims;
        _mega_block_sizes.set_vals_same(0);   // 0 => rank size.
        _mega_block_tile_sizes = dims->_stencil_dims;
        _mega_block_tile_sizes.set_vals_same(0); // 0 => mega-block size.
        
        _block_sizes = dims->_stencil_dims;
        _block_sizes.set_vals_same(0); // 0 => mega-block size.
        _block_tile_sizes = dims->_stencil_dims;
        _block_tile_sizes.set_vals_same(0); // 0 => block size.

        _micro_block_sizes = dims->_stencil_dims;
        _micro_block_sizes.set_vals_same(0); // 0 => block size.
        _micro_block_tile_sizes = dims->_stencil_dims;
        _micro_block_tile_sizes.set_vals_same(0); // 0 => micro-block size.

        _nano_block_sizes = dims->_stencil_dims;
        _nano_block_sizes.set_vals_same(0);      // 0 => micro-block size.
        _nano_block_tile_sizes = dims->_stencil_dims;
        _nano_block_tile_sizes.set_vals_same(0); // 0 => nano-block size.

        _pico_block_sizes = dims->_stencil_dims;
        _pico_block_sizes.set_vals_same(0);            // 0 => cluster size.

        _min_pad_sizes = dims->_stencil_dims;
        _min_pad_sizes.set_vals_same(0);

        _extra_pad_sizes = dims->_stencil_dims;
        _extra_pad_sizes.set_vals_same(0);

        // Use only domain dims for MPI tuples.
        _num_ranks = dims->_domain_dims;
        _num_ranks.set_vals_same(0); // 0 => set using heuristic.

        _rank_indices = dims->_domain_dims;
        _rank_indices.set_vals_same(0);

        // Things to tune.
        #ifdef USE_OFFLOAD
        _tuner_targets.push_back(_pico_block_str);
        #else
        _tuner_targets.push_back(_block_str);
        #endif
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
            parser.add_option(make_shared<CommandLineParser::IdxOption>
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
        parser.add_option(make_shared<CommandLineParser::MultiIdxOption>
                          (shortcut,
                           "Shortcut for" + multi_help,
                           multi_vars));
    }

    // Add access to these options from a cmd-line parser.
    void KernelSettings::add_options(CommandLineParser& parser)
    {
        // Following options are in the 'yask' namespace, i.e., no object.
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("print_suffixes",
                           "Format output with suffixes for human readibility, e.g., 6.15K, 12.3GiB, 7.45m."
                           " If disabled, prints without suffixes for computer parsing, e.g., 6150, 1.23e+10, 7.45e-3.",
                           yask::is_suffix_print_enabled));

        // Following options are in 'this' object.
        _add_domain_option(parser, "g", "Global-domain (overall-problem) size", _global_sizes);
        _add_domain_option(parser, "l", "Local-domain (rank) size", _rank_sizes);
        _add_domain_option(parser, _mega_block_str, "Mega-block size", _mega_block_sizes, true);
        _add_domain_option(parser, _block_str, "Block size", _block_sizes, true);
        _add_domain_option(parser, _micro_block_str, "Micro-block size", _micro_block_sizes);
        _add_domain_option(parser, _nano_block_str, "Nano-block size", _nano_block_sizes);
        _add_domain_option(parser, _pico_block_str, "Pico-block size", _pico_block_sizes);
        _add_domain_option(parser, "d", "[Deprecated] Use local-domain size options", _rank_sizes);
        #ifdef USE_TILING
        _add_domain_option(parser, "l_tile", "[Advanced] Local-domain-tile size", _rank_tile_sizes);
        _add_domain_option(parser, "Mb_tile", "[Advanced] Mega-Block-tile size", _mega_block_tile_sizes);
        _add_domain_option(parser, "b_tile", "[Advanced] Block-tile size", _block_tile_sizes);
        _add_domain_option(parser, "mb_tile", "[Advanced] Micro-block-tile size", _micro_block_tile_sizes);
        _add_domain_option(parser, "nb_tile", "[Advanced] Nano-block-tile size", _nano_block_tile_sizes);
        #endif
        _add_domain_option(parser, "mp", "[Advanced] Minimum padding size (including halo)"
                           " applied to all YASK vars", _min_pad_sizes);
        _add_domain_option(parser, "ep", "[Advanced] Extra padding size (beyond halo)"
                           " applied to all YASK vars", _extra_pad_sizes);
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("allow_addl_padding",
                           "[Advanced] Allow automatic extension of padding for"
                           " additional performance on any or all YASK vars.",
                           _allow_addl_pad));
        #ifdef USE_MPI
        _add_domain_option(parser, "nr", "Num ranks", _num_ranks);
        _add_domain_option(parser, "ri", "This rank's logical index (0-based)", _rank_indices);
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("overlap_comms",
                           "Overlap MPI communication with calculation of interior elements whenever possible.",
                           overlap_comms));
        parser.add_option(make_shared<CommandLineParser::IdxOption>
                          ("min_exterior",
                           "[Advanced] Minimum width of exterior section to"
                           " compute before starting MPI communication. "
                           "Applicable only when overlap_comms is enabled.",
                           _min_exterior));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("exchange_halos",
                           "[Debug] Perform halo packs/unpacks/sends/receives. "
                           "Will not give correct results if disabled.",
                           do_halo_exchange));
        #ifdef USE_OFFLOAD
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("use_device_mpi",
                           "Enable device-to-device MPI transfers using device addresses. "
                           "Must be supported by MPI library and hardware.",
                           use_device_mpi));
        #else
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("use_shm",
                           "Directly use shared memory for halo-exchange buffers "
                           "between ranks on the same node when possible. "
                           "Otherwise, use the same non-blocking MPI send and receive calls "
                           "that are used between nodes.",
                           use_shm));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("force_scalar_exchange",
                           "[Debug] Do not allow vectorized halo exchanges.",
                           force_scalar_exchange));
        #endif
        #endif
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("force_scalar",
                           "[Debug] Evaluate every var point with scalar stencil operations "
                           "and exchange halos using only scalar packing and unpacking.",
                           force_scalar));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("max_threads",
                           "Maximum number of OpenMP CPU threads to use for both outer and inner threads. "
                           "If zero (0), the default value from the OpenMP library is used.",
                           max_threads));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("outer_threads",
                           "Number of CPU threads to use in the outer OpenMP region. "
                           "Specifies how many blocks may be executed concurrently within each mega-block. "
                           "Will be restricted to a value less than or equal to "
                           "the maximum number of OpenMP threads specified by -max_threads "
                           "divided by the number specified by -inner_threads. "
                           "If zero (0), set to the value specified by -max_threads "
                           "divided by the number specified by -inner_threads.",
                           num_outer_threads));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("inner_threads",
                           "Number of CPU threads to use in each inner (nested) OpenMP region. "
                           "Specifies how many nano-blocks may be executed concurrently within each micro-block. "
                           "Will be restricted to a value less than or equal to "
                           "the maximum number of OpenMP threads specified by -max_threads. "
                           "If zero (0), set to one (1).",
                           num_inner_threads));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("block_threads",
                           "[Deprecated] Use 'inner_threads' option.",
                           num_inner_threads));
        #ifdef USE_OFFLOAD
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("device_thread_limit",
                           "Set the maximum number of OpenMP device threads used within a team.",
                           thread_limit));
        #endif
        #ifndef USE_OFFLOAD
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("bind_inner_threads",
                           "[Advanced] Divide micro-blocks into nano-blocks of slabs along the first valid dimension "
                           "(usually the outer-domain dimension), ignoring other nano-block sizes. "
                           "Assign each slab to an inner thread based on its global index in that dimension. "
                           "This setting may increase cache locality when using more than one "
                           "inner thread, especially when scratch vars are used and/or "
                           "when temporal blocking is active. "
                           "This option is ignored if there are fewer than two inner threads.",
                           bind_inner_threads));
        #endif
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("bundle_allocs",
                           "[Advanced] Allocate memory for multiple YASK vars in "
                           "a single large chunk when possible. "
                           "If 'false', allocate each YASK var separately.",
                           _bundle_allocs));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("numa_pref",
                           string("[Advanced] Specify allocation policy for vars and MPI buffers. ") +
                           #ifdef USE_NUMA
                           " Use values >= 0 to specify the preferred NUMA node. "
                           " Use " + to_string(yask_numa_local) + " for local NUMA-node allocation. " +
                           " Use " + to_string(yask_numa_interleave) + " for interleaving pages across NUMA nodes. " +
                           #endif
                           #ifdef USE_OFFLOAD
                           " Use " + to_string(yask_numa_offload) + " for allocation optimized for offloading. " +
                           #endif
                           " Use " + to_string(yask_numa_none) + " for default allocator.",
                           _numa_pref));
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("auto_tune",
                           "Adjust specified block and tile sizes *during* normal operation "
                           "to tune for performance, i.e., 'online' or 'in-situ' tuning. "
                           "Successive steps will use different sizes until an optimal size is found. "
                           "Will likely cause varying performance between steps, "
                           "so this is not recommended for benchmarking. "
                           "However, this can be a useful tool for deployment of YASK stencils "
                           "onto unknown and/or varied systems where 'offline' tuning is not practical.",
                           _do_auto_tune));
        parser.add_option(make_shared<CommandLineParser::DoubleOption>
                          ("auto_tune_trial_secs",
                           "[Advanced] Seconds to run new trial during auto-tuning "
                           "for new trial to be considered better than the existing best.",
                           _tuner_trial_secs));
        parser.add_option(make_shared<CommandLineParser::IntOption>
                          ("auto_tune_radius",
                           "[Advanced] Starting search radius for tuning block sizes. "
                           "A power of 2 is recommended.",
                           _tuner_radius));
        #ifdef ALLOW_STAGE_TUNERS
        parser.add_option(make_shared<CommandLineParser::BoolOption>
                          ("auto_tune_each_stage",
                           "[Advanced] Apply the auto-tuner separately to each stage. "
                           "Will only be used if stages are applied in separate "
                           "passes across the entire grid, "
                           "i.e., when no temporal tiling is used.",
                           _allow_stage_tuners));
        #endif

        // Make set of allowed auto-tune targets.
        set<string> allowed_targets;
        allowed_targets.insert(_mega_block_str);
        allowed_targets.insert(_block_str);
        allowed_targets.insert(_micro_block_str);
        allowed_targets.insert(_nano_block_str);
        allowed_targets.insert(_pico_block_str);
        parser.add_option(make_shared<CommandLineParser::StringListOption>
                          ("auto_tune_targets",
                           "[Advanced] Apply the auto-tuner to adjust the sizes of the listed targets. "
                           "Allowed targets are "
                           "'" + _mega_block_str + "' for mega-block sizes, "
                           "'" + _block_str + "' for block sizes, "
                           "'" + _micro_block_str + "' for micro-block sizes, "
                           "'" + _nano_block_str + "' for nano-block sizes, and "
                           "'" + _pico_block_str + "' for pico-block sizes. "
                           "Targets must be separated by a single comma (','). "
                           "Targets will be tuned in the order given and may be repeated.",
                           allowed_targets, _tuner_targets));
    }

    // Print usage message.
    void KernelSettings::print_usage(ostream& os)
    {
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
            "  Ranks may be evaluated in parallel in separate MPI processes.\n"
            "  The purpose of local-domains is to control the amount of work done in one\n"
            "   entire MPI rank.\n"
            "  Each local-domain is composed of one or more mega-blocks.\n"
            " A 'mega-block' is a sub-division of work within a local-domain.\n"
            "  Mega-blocks are evaluated sequentially within ranks.\n"
            "  The purpose of mega-blocks is to control the amount of work done across an\n"
            "   entire MPI rank while sharing a large cache.\n"
            "  If using temporal wave-front rank tiling (see mega-block-size guidelines),\n"
            "   then this is the work done in each wave-front rank tile;\n"
            "   else, there is typically only one mega-block the size of the local-domain.\n"
            "  Each mega-block is composed of one or more blocks.\n"
            " A 'block' is a sub-division of work within a mega-block.\n"
            "  Blocks may be evaluated in parallel within mega-blocks.\n"
            "  The purpose of blocking is to provide control over the amount of\n"
            "   work done by each outer OpenMP thread.\n"
            "  This is the most commonly-tuned work-size for many stencils, especially\n"
            "   when not using any sort of temporal tiling.\n"
            "  Each block is composed of one or more micro-blocks.\n"
            " A 'micro-block' is a sub-division of work within a block.\n"
            "  Micro-blocks are evaluated sequentially within blocks.\n"
            "  The purpose of micro-blocking is to allow distinction between the amount\n"
            "   of work done by an outer thread (via blocking) and the amount of work done\n"
            "   for cache locality (via micro-blocking).\n"
            "  If using temporal wave-front block tiling (see micro-block-size guidelines),\n"
            "   then this is the work done for each wave-front block tile,\n"
            "   and the number temporal steps in the micro-block is always equal\n"
            "   to the number temporal steps a block;\n"
            "   else, there is typically only one micro-block the size of a block.\n"
            "  If micro-block sizes are not specified, a micro-block is the same size\n"
            "   as a block, and the amount of work done by a thread and the amount of\n"
            "   work done for cache locality is the same.\n"
            "  Each micro-block is composed of one or more nano-blocks.\n"
            " A 'nano-block' is a sub-division of work within a micro-block.\n"
            "  Nano-blocks may be evaluated in parallel within micro-blocks.\n"
            "  The purpose of nano-blocking is to allow multiple inner OpenMP threads\n"
            "   to cooperatively work on a micro-block, sharing cached values--\n"
            "   this is particularly useful when using hyper-threads on a CPU.\n"
            "  If the number of inner OpenMP threads is greater than one,\n"
            "   then this is the unit of work for each nested thread,\n"
            "   and nano-blocks are evaluated concurrently within each micro-block;\n"
            "   else, nano-blocks are evaluated sequentially within each micro-block.\n"
            #ifdef USE_OFFLOAD
            "  When offloading computation to a device, a nano-block is the unit of work\n"
            "   done in each offloaded kernel invocation.\n"
            #endif
            "  There is no temporal tiling at the nano-block level.\n"
            "  Each nano-block is composed of one or more pico-blocks.\n"
            " A 'pico-block' is a sub-division of work within a nano-block.\n"
            #ifdef USE_OFFLOAD
            " Pico-blocks may be evaluated in parallel within nano-blocks on the device.\n"
            #else
            " Pico-blocks are evaluated sequentially within nano-blocks.\n"
            #endif
            "  The purpose of a pico-block is to allow additional cache-locality\n"
            "   at this low level.\n"
            #ifdef USE_OFFLOAD
            "  When offloading computation to a device, a pico-block allows\n"
            "   cache-locality within a kernel work item.\n"
            #endif
            "  There is no temporal tiling at the pico-block level.\n"
            "  Each pico-block is composed of one or more clusters.\n"
            " A 'cluster' is the work done in each inner-most pico-loop iteration.\n"
            "  Clusters are evaluated sequentially within pico-blocks.\n"
            "  The purpose of clustering is to allow more than one vector of\n"
            "   work to be done in each loop iteration, useful for very simple stencils.\n"
            "  Each cluster is composed of one or more vectors.\n"
            " A 'vector' is typically the work done by a SIMD instruction.\n"
            "  Vectors are evaluated sequentially within clusters.\n"
            "  A 'folded vector' contains points in more than one dimension.\n"
            "  The size of a vector is typically that of a SIMD register.\n"
            "  Each vector is composed of one or more points.\n"
            " A 'point' is a single floating-point (FP) element in a grid.\n"
            "  Points may be evaluated in parallel within vectors.\n"
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
            " The mega-block sizes are used to configure temporal wave-front rank tiling.\n"
            "  Temporal wave-front rank tiling may increase locality in large shared caches\n"
            "   when a local-domain is larger than the capacity of those caches.\n"
            "  A mega-block size >1 in the step dimension (e.g., '-rt') enables wave-front rank tiling.\n"
            "  A mega-block size of 0 in the step dimension => the temporal wave-front\n"
            "   rank tiling will have the same number of steps as the temporal block tiling.\n"
            "  The mega-block size in the step dimension affects how often MPI halo-exchanges occur:\n"
            "   A mega-block size of 0 in the step dimension => exchange after every stage.\n"
            "   A mega-block size >0 in the step dimension => exchange after that many steps.\n"
            " The block sizes specify the work done by each top-level OpenMP thread.\n"
            "  A block size of 0 in a given domain dimension =>\n"
            "   block size is set to mega-block size in that dimension.\n"
            "  A block size of 0 in the step dimension (e.g., '-bt') disables any temporal blocking.\n"
            "  A block size of 1 in the step dimension enables temporal blocking, but only between\n"
            "   stages in the same step.\n"
            "  A block size >1 in the step dimension enables temporal mega-block tiling across multiple steps.\n"
            "  The temporal block size may be automatically reduced if needed based on the\n"
            "   domain block sizes, the stencil halos, and the step size of the mega-blocks.\n"
            " The micro-block sizes are used to configure temporal wave-front block tiling.\n"
            "   Temporal wave-front block tiling may increase locality in core-local caches\n"
            "   (e.g., L2) when blocks are larger than that the capacity of those caches.\n"
            "  A micro-block size of 0 in a given domain dimension =>\n"
            "   micro-block size is set to block size in that dimension.\n"
            "  The size of a micro-block in the step dimension is always implicitly\n"
            "   the same as that of a block.\n"
            " The nano-block sizes specify the work done by each nested OpenMP thread.\n"
            "  Multiple nano-blocks may enable more effective sharing of caches\n"
            "   among multiple hyper-threads in a core when there is more than\n"
            "   one block-thread. It can also be used to share data between caches\n"
            "   among multiple cores.\n"
            "  A nano-block size of 0 in a given domain dimension =>\n"
            "   nano-block size is set to micro-block size in that dimension;\n"
            "   when there is more than one block-thread, the first dimension\n"
            "   will instead be set to the vector length to create \"slab\" shapes.\n"
            "  A pico-block size of 0 in a given domain dimension =>\n"
            "   pico-block size is set to cluster size in that dimension;\n"
            " The vector and cluster sizes are set at compile-time, so\n"
            "  there are no run-time options to set them.\n"
            #ifdef USE_TILING
            " Set 'tile' sizes to provide finer control over the order of evaluation\n"
            "  within the given area. For example, nano-block-tiles create smaller areas\n"
            "  within nano-blocks; points with the first nano-block-tile will be scheduled\n"
            "  before those the second nano-block-tile, etc. (There is no additional level\n"
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
            "\nControlling OpenMP CPU threading:\n"
            " For stencil evaluation, threads are allocated using nested OpenMP:\n"
            "  Num outer_threads = max_threads / inner_threads if not specified.\n"
            "  Num CPU threads per rank and mega-block = outer_threads.\n"
            "  Num CPU threads per block = inner_threads.\n"
            "  Num CPU threads per micro-block, nano-block, and pico-block = 1.\n";
    }
    void KernelSettings::print_values(ostream& os)
    {
        CommandLineParser soln_parser;
        add_options(soln_parser);
        soln_parser.print_values(os);
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

        // Null stream to throw away debug info if 'ksb' is null.
        yask_output_factory yof;
        auto nullop = yof.new_null_output();
        yask_output_ptr op = ksb ? ksb->get_debug_output() : nullop;
        ostream& os = op->get_ostream();

        auto& step_dim = _dims->_step_dim;
        auto& inner_layout_dim = _dims->_inner_layout_dim;
        auto& inner_loop_dim = _dims->_inner_loop_dim;
        auto& rt = _mega_block_sizes[step_dim];
        auto& bt = _block_sizes[step_dim];
        auto& mbt = _micro_block_sizes[step_dim];
        auto& cluster_pts = _dims->_cluster_pts;
        int nddims = _dims->_domain_dims.get_num_dims();

        // Fix up step-dim sizes.
        rt = max(rt, idx_t(0));
        bt = max(bt, idx_t(0));
        mbt = max(mbt, idx_t(0));
        if (!rt)
            rt = bt;       // Default mega-block steps == block steps.
        if (!mbt)
            mbt = bt;       // Default micro-blk steps == block steps.

        // Adjust defaults for blocks on CPU or pico-blocks on GPU.
        DOMAIN_VAR_LOOP(i, j) {
            #ifdef USE_OFFLOAD
            if (!_pico_block_sizes[i])
                _pico_block_sizes[i] = def_blk_size;
            #else
            if (!_block_sizes[i])
                _block_sizes[i] = def_blk_size;
            #endif
        }
        
        // Determine num mega-blocks.
        // Also fix up mega-block sizes as needed.
        // Temporal mega-block size will be increase to
        // current temporal block size if needed.
        // Default mega-block size (if 0) will be size of rank-domain.
        os << "\nMega-Blocks:" << endl;
        auto nr = find_num_subsets(os,
                                   _mega_block_sizes, "mega-block",
                                   _rank_sizes, "local-domain",
                                   cluster_pts, "cluster",
                                   step_dim);
        os << " num-mega-blocks-per-local-domain-per-step: " << nr << endl;
        os << " Since the mega-block size in the '" << step_dim <<
            "' dim is " << rt << ", temporal wave-front tiling of each local-domain is ";
        if (!rt) os << "NOT ";
        os << "enabled.\n";

        // Determine num blocks.
        // Also fix up block sizes as needed.
        os << "\nBlocks:" << endl;
        auto nb = find_num_subsets(os,
                                   _block_sizes, "block",
                                   _mega_block_sizes, "mega-block",
                                   cluster_pts, "cluster",
                                   step_dim);
        os << " num-blocks-per-mega-block-per-step: " << nb << endl;
        os << " num-blocks-per-local-domain-per-step: " << (nb * nr) << endl;
        os << " Since the block size in the '" << step_dim <<
            "' dim is " << bt << ", temporal concurrent tiling of each mega-block is ";
        if (!bt) os << "NOT ";
        os << "enabled.\n";

        // Determine num micro-blocks.
        // Also fix up micro-block sizes as needed.
        os << "\nMicro-blocks:" << endl;
        auto nmb = find_num_subsets(os,
                                    _micro_block_sizes, "micro-block",
                                    _block_sizes, "block",
                                    cluster_pts, "cluster",
                                    step_dim);
        os << " num-micro-blocks-per-block-per-step: " << nmb << endl;
        os << " num-micro-blocks-per-mega-block-per-step: " << (nmb * nb) << endl;
        os << " num-micro-blocks-per-local-domain-per-step: " << (nmb * nb * nr) << endl;
        os << " Since the micro-block size in the '" << step_dim <<
            "' dim is " << mbt << ", temporal wave-front tiling of each block is ";
        if (!mbt) os << "NOT ";
        os << "enabled.\n";

        // Adjust defaults for nano-blocks to be slab if we are using more
        // than one block thread.  Otherwise, find_num_subsets() would set
        // default to entire block, and we wouldn't use multiple threads.
        if (num_inner_threads > 1 && _nano_block_sizes.sum() == 0) {

            // Default dim is outer one.
            _bind_posn = 1;

            // Look for best dim to split and bind threads to
            // if binding is enabled.
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.get_dim_name(j);
                if (dname == inner_loop_dim)
                    continue;

                auto bsz = _block_sizes[i];
                auto cpts = cluster_pts[j];
                auto clus_per_blk = bsz / cpts;

                // Subdivide this dim if there are enough clusters in
                // the block for each thread.
                if (clus_per_blk >= num_inner_threads) {
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
                _nano_block_sizes[_bind_posn] = cpts;

            // Divide block equally.
            else
                _nano_block_sizes[_bind_posn] = ROUND_UP(bsz / num_inner_threads, cpts);
        }

        // Determine num nano-blocks.
        // Also fix up nano-block sizes as needed.
        os << "\nNano-blocks:" << endl;
        auto nsb = find_num_subsets(os,
                                    _nano_block_sizes, "nano-block",
                                    _micro_block_sizes, "micro-block",
                                    cluster_pts, "cluster",
                                    step_dim);
        os << " num-nano-blocks-per-micro-block-per-step: " << nsb << endl;
        os << " num-nano-blocks-per-block-per-step: " << (nsb * nmb) << endl;
        os << " num-nano-blocks-per-mega-block-per-step: " << (nsb * nmb * nb) << endl;
        os << " num-nano-blocks-per-rank-per-step: " << (nsb * nmb * nb * nr) << endl;
        os << " Temporal tiling of micro-blocks is never enabled.\n";
        
        // Determine num pico-blocks.
        // Also fix up pico-block sizes as needed.
        os << "\nPico-blocks:" << endl;
        auto npb = find_num_subsets(os,
                                    _pico_block_sizes, "pico-block",
                                    _nano_block_sizes, "nano-block",
                                    cluster_pts, "cluster",
                                    step_dim);
        os << " num-pico-blocks-per-nano-block-per-step: " << npb << endl;
        os << " num-pico-blocks-per-micro-block-per-step: " << (npb * nsb) << endl;
        os << " num-pico-blocks-per-block-per-step: " << (npb * nsb * nmb) << endl;
        os << " num-pico-blocks-per-mega-block-per-step: " << (npb * nsb * nmb * nb) << endl;
        os << " num-pico-blocks-per-rank-per-step: " << (npb * nsb * nmb * nb * nr) << endl;
        os << " Temporal tiling of nano-blocks is never enabled.\n";

        // Determine binding dimension. Do this again if it was done above
        // by default because it may have changed during adjustment.
        if (bind_inner_threads && num_inner_threads > 1) {
            DOMAIN_VAR_LOOP(i, j) {

                // Don't pick inner dim.
                auto& dname = _dims->_domain_dims.get_dim_name(j);
                if (dname == inner_loop_dim)
                    continue;

                auto bsz = _block_sizes[i];
                auto sbsz = _nano_block_sizes[i];
                auto sb_per_b = CEIL_DIV(bsz, sbsz);

                // Choose first dim with enough nano-blocks
                // per block.
                if (sb_per_b >= num_inner_threads) {
                    _bind_posn = i;
                    break;
                }
            }
            os << " Note: only the nano-block size in the '" <<
                _dims->_stencil_dims.get_dim_name(_bind_posn) << "' dimension may be used at run-time\n"
                "  because block-thread binding is enabled on " << num_inner_threads << " block threads.\n";
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
                                    _mega_block_sizes, "mega-block",
                                    step_dim);
        os << " num-local-domain-tiles-per-local-domain-per-step: " << nlg << endl;

        // Show num mega-block-tiles.
        // TODO: only print this if mega-block-tiling is enabled.
        os << "\nMega-Block tiles:\n";
        auto nrg = find_num_subsets(os,
                                    _mega_block_tile_sizes, "mega-block-tile",
                                    _mega_block_sizes, "mega-block",
                                    _block_sizes, "block",
                                    step_dim);
        os << " num-mega-block-tiles-per-mega-block-per-step: " << nlg << endl;

        // Show num block-tiles.
        // TODO: only print this if block-tiling is enabled.
        os << "\nBlock tiles:\n";
        auto nbg = find_num_subsets(os,
                                    _block_tile_sizes, "block-tile",
                                    _block_sizes, "block",
                                    _micro_block_sizes, "micro-block",
                                    step_dim);
        os << " num-block-tiles-per-block-per-step: " << nbg << endl;

        // Show num micro-block-tiles.
        // TODO: only print this if micro-block-tiling is enabled.
        os << "\nMicro-block tiles:\n";
        auto nmbt = find_num_subsets(os,
                                     _micro_block_tile_sizes, "micro-block-tile",
                                     _micro_block_sizes, "micro-block",
                                     _nano_block_sizes, "nano-block",
                                     step_dim);
        os << " num-micro-block-tiles-per-micro-block-per-step: " << nmbt << endl;

        // Show num nano-block-tiles.
        // TODO: only print this if nano-block-tiling is enabled.
        os << "\nNano-block tiles:\n";
        auto nsbt = find_num_subsets(os,
                                     _nano_block_tile_sizes, "nano-block-tile",
                                     _nano_block_sizes, "nano-block",
                                     _pico_block_sizes, "pico-block",
                                     step_dim);
        os << " num-nano-block-tiles-per-nano-block-per-step: " << nsbt << endl;

        // NB: there are no pico-block tiles.
        #endif
        os << endl;
    }

    // Ctor.
    KernelStateBase::KernelStateBase(KernelEnvPtr& kenv,
                                     KernelSettingsPtr& kactl_opts,
                                     KernelSettingsPtr& kreq_opts)
    {
        host_assert(kenv);
        host_assert(kactl_opts);
        host_assert(kreq_opts);
        host_assert(kactl_opts->_dims);

        // Create state. All other objects that need to share
        // this state should use a shared ptr to it.
        _state = make_shared<KernelState>();
       
        // Share passed ptrs.
        _state->_env = kenv;
        _state->_actl_opts = kactl_opts;
        _state->_req_opts = kreq_opts;
        _state->_dims = kactl_opts->_dims;

        // Create MPI Info object.
        _state->_mpi_info = make_shared<MPIInfo>(_state->_dims);

        // Set vars after above inits.
        STATE_VARS(this);

    }

    // Set number of threads w/o using thread-divisor.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_max_threads() {
        STATE_VARS(this);

        // Get max number of threads.
        int mt = max(actl_opts->max_threads, 1);

        // Set num threads to use for inner and outer loops.
        yask_num_threads[0] = mt;
        yask_num_threads[1] = 0;

        // Reset number of OMP threads to max allowed.
        omp_set_num_threads(mt);
        return mt;
    }

    // Get total number of computation threads to use.
    int KernelStateBase::get_num_comp_threads(int& outer_threads, int& inner_threads) const {
        STATE_VARS(this);

        int mt = max(actl_opts->max_threads, 1);

        int it = max(actl_opts->num_inner_threads, 1);
        it = min(it, mt);
        inner_threads = it;

        int max_ot = max(mt / it, 1);
        int ot = actl_opts->num_outer_threads;
        if (ot <= 0)
            ot = max_ot;
        ot = min(ot, max_ot);
        outer_threads = ot;

        // Total number of inner threads.
        // Might be less than max threads due to truncation.
        int ct = it * ot;
        assert(ct <= mt);
        return ct;
    }

    // Set number of threads to use for a mega-block.
    // Enable nested OMP.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_num_outer_threads() {
        int rt=0, bt=0;
        int at = get_num_comp_threads(rt, bt);

        // Must call before entering top parallel mega-block.
        int ol = omp_get_level();
        assert(ol == 0);

        // Enable nested OMP.
        omp_set_nested(1);
        omp_set_max_active_levels(yask_max_levels + 1); // Add 1 for offload.
         
        // Set num threads to use for inner and outer loops.
        yask_num_threads[0] = rt;
        yask_num_threads[1] = bt;

        // Set num threads for a mega-block.
        omp_set_num_threads(rt);
        return rt;
    }

    // Set number of threads for a block.
    // Must be called from within a top-level OMP parallel mega-block.
    // Return number of threads.
    // Do nothing and return 0 if not properly initialized.
    int KernelStateBase::set_num_inner_threads() {
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
