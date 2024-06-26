# YASK--Yet Another Stencil Kit

* New YASK users may want to start with
the [YASK tutorial](http://intel.github.io/yask/YASK-tutorial.pdf).
* Users with existing YASK-based code may want to jump to
the [backward-compatibility notices](#backward-compatibility-notices).
* All YASK users will also be interested in
the [API documentation](http://intel.github.io/yask/api/html/index.html).

## Overview
YASK is a framework to rapidly create high-performance stencil code
including optimizations and features such as
* Support for boundary layers and staggered-grid stencils.
* Vector-folding to increase data reuse via non-traditional data layout.
* Multi-level OpenMP parallelism to exploit multiple CPU cores and threads.
* OpenMP offloading to GPUs.
* MPI scaling to multiple sockets and nodes with overlapped communication and compute.
* Spatial tiling with automatically-tuned block sizes.
* Temporal tiling in multiple dimensions to further increase cache locality.
* APIs for C++ and Python.

YASK contains a domain-specific compiler to convert stencil-equation specifications to
optimized code for Intel(R) Xeon(R) processors, Intel(R) Xeon Phi(TM) processors,
and Intel(R) graphics processors.

### Supported Platforms and Processors:
* 64-bit Linux.
* Intel(R) Xeon(R) processors supporting the AVX, AVX2, or CORE_AVX512 instruction sets.
* Intel(R) Xeon Phi(TM) x200-family processors supporting the MIC_AVX512 instruction set (KNL).
* Intel(R) graphics processors supporting UHD graphics, e.g., Intel(R) Data Center GPU Max Series products.

### Pre-requisites:
* Intel(R) [oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html)
  HPC Toolkit for Linux (toolkit 2024.1 or later recommended); this will install
  the Intel(R) oneAPI DPC++/C++ Compiler and the Intel(R) MPI Library.
  See compiler notes below under version 4.00.00 changes.
* Gnu C++ compiler, g++ (8.5.0 or later recommended).
  (Even when using Intel compilers, a g++ installation is required.)
* Linux libraries `librt` and `libnuma`.
* Grep.
* Perl (v5 or later).
* Awk.
* Gnu make.
* Bash shell.
* Numactl utility if running on more than one CPU socket.
* Optional utilities and their purposes:
    * The `indent` or `gindent` utility, used automatically during the build process
      to make the generated code easier for humans to read.
      You'll get a warning when running `make` if one of these doesn't exist.
      Everything will still work, but the generated code will be difficult to read.
      Reading the generated code is only necessary for debug, performance analysis, etc.
    * SWIG (4.0.0 or later):
      http://www.swig.org, for creating the Python interface.
    * Python 3 (3.6.1 or later):
      https://www.python.org/downloads, for creating and using the Python interface.
      Included with Intel(R) oneAPI HPC Toolkit.
    * Python `numpy` package for running Python interface tests.
      Included with Intel(R) oneAPI HPC Toolkit.
    * Doxygen (1.9.0 or later):
      https://www.doxygen.nl, for creating updated API documentation.
      If you're not changing the API documentation, you can view the existing documentation
      at the link at the top of this page.
    * Graphviz (2.30.1 or later):
      http://www.graphviz.org, for rendering stencil diagrams.
    * Intel(R) Software Development Emulator:
      https://software.intel.com/en-us/articles/intel-software-development-emulator,
      for functional testing if you don't have native support for the targeted instruction set.

## Backward-compatibility notices
### Version 4
* Version 4.05.03 has a few notices:
  - The default stencil name "iso3dfd" is removed; this means
    you must always set the stencil name when building a kernel -- there is no default.
    If you get an error about the "unspecified" stencil, this means you didn't set the
    stencil name.
  - Some of the API-related Makefile target names were changed for consistency.
* Version 4.05.00 removes the "out-of-band" genetic-algorithm tuning script
    due to lack of resources for maintenance and testing.
* Version 4.04.00 deprecates the existing `void* {set,get}_elements_in_slice()`
    APIs and provides safer `float*` and `double*` versions.
* Version 4.03.00 is a significant release with the following notices:
  - Each non-scratch stencil equation is now checked to ensure
    offsets of +/-1 from the step-dimension on the LHS, e.g.,
    `A(t+1, x, y) EQUALS B(t, x, y+1)`.
    (-1 is used for less-common reverse-time stencils.)
  - The `yk_solution::get_var()` API now throws an exception if the
    named var does not exist. (Used to return `std::nullptr`.)
  - Vector "clustering" (unrolling by the YASK compiler) is no
    longer supported.
  - Read-ahead in the inner-loop is no longer supported.
  - APIs for getting OpenMP thread counts were added.
  - Equation "bundles" are now called solution "parts".
* Version 4.01.00 added several new APIs.
    The following changes were made to to the YASK compiler:
    removed the `-eq_bundles` option, and
    an exception is now thrown from `output_solution()` if the
    format string is unrecognized.
* Version 4.00.00 was a major release with a number of notices:
  - Support has been added for GPU offloading via the OpenMP device model.
    Build any YASK stencil kernel with `make offload=1 ...`. This will create
    a kernel library and executable with an "arch" field containing
    "offload" and the OpenMP device target name.
    Use `make offload=1 offload_arch=<target>` to change the OpenMP target;
    the default is `spir64`, for GPUs with Intel(R) Architecture (e.g., Gen12).
    Use `make offload_usm=1` to use the OpenMP Unified Shared Memory model.
  - The default compiler is now the Intel(R) oneAPI C++ compiler, icpx.
    If you want to use a different compiler, use `make YK_CXX=<compiler> ...`
    for the kernel, and/or `make YC_CXX=<compiler> ...` for the YASK compiler,
    or `make CXX=<compiler>` for both. A C++ compiler that supports C++17
    is now required.
  - The loop hierarchy has been extended and renamed with (hopefully)
    more memorable names:
    version 3's regions, blocks, mini-blocks, and sub-blocks
    are now mega-blocks, blocks, micro-blocks, and nano-blocks,
    respectively.
    Pico-blocks have been added inside nano-blocks.
    When offloading, the nano-blocks and pico-blocks are executed on the device.
    The looping behaviors, including any temporal tiling, of mega-blocks,
    blocks, and micro-blocks are handled by the CPU.
    The `get_region_size()` and `set_region_size()` APIs have been removed.
    The `-r` and `-sb` options, e.g., `-rx` and `-sbx`, have also been removed.
  - Regarding CPU threads, "region threads" are now referred to as "outer threads",
    and "block threads" are now referred to as "inner threads".
    The option `-block_threads` is deprecated.
    The option `-thread_divisor` has been removed.
    See the `-help` documentation for new options `-outer_threads` and `-inner_threads`.
    The `-max_threads` option remains.
  - Only one thread per core is now used by default on most CPU models.
    This is done in `yask.sh` by passing `-outer_threads <N>` to the executable,
    where `<N>` is the number of cores on the node divided by the
    number of MPI ranks.
    Consequently, the default number of inner threads is now one (1)
    to use one core per block.
    This change was made based on observed
    performance on newer Intel(R) Xeon(R) Processors.  Previous versions
    used two threads per block by default and used both hyper-threads if
    they were enabled.  To configure two hyper-threads to work cooperatively
    on each block, use the option `-inner_threads 2`.
    These changes do not
    apply to Intel(R) Xeon Phi(TM) x200-family processors (KNL), which
    continue to use all 4 hyper-threads per core and 8 inner threads
    by default (because 2 cores share an L2 cache).
  - Intel(R) Xeon Phi(TM) x100-family processors (KNC) are no longer supported.
    (Intel(R) Xeon Phi(TM) x200-family processors (KNL) are still supported.)
  - Python v2 is no longer supported.
  - New vector APIs were added to `yk_solution` and `yk_var` to allow getting
    or setting multiple dimensions in one API call.
  - `new_relative_var_point()` API is deprecated.
  - APIs that were previously deprecated in the `yk_var` class have been removed.
  - Explicit support for persistent-memory devices has been removed.
    (Persistent-memory accessible via separate NUMA nodes or other standard
    Linux mechanisms is supported as with any other special memory types,
    e.g., high-bandwidth memory.)

### Version 3
* Version 3.05.00 changed the default setting of `-use_shm` to `true`.
  Use `-no-use_shm` to disable shared-memory inter-rank communication.
* Version 3.04.00 changed the terms "pack" and "pass" to "stage", which may affect
  user-written result parsers. Option `auto_tune_each_pass` changed to
  `auto_tune_each_stage`.
* Version 3.01.00 moved the `-trace` and `-msg_rank` options from the kernel
  library to the kernel utility, so those options may no longer be set via
  `yk_solution::apply_command_line_options()`. APIs to set the corresponding
  options are now in `yk_env`. This allows configuring the debug output
  before a `yk_solution` is created.
* Version 3.00.00 was a major release with a number of notices:
  - The old (v1 and v2) internal DSL that used undocumented types such as
    `SolutionBase` and `GridValue` and undocumented macros such as
    `MAKE_GRID` was replaced with an expanded version of the documented YASK
    compiler API.  Canonical v2 DSL code should still work using the
    `Soln.hpp` backward-compatibility header file.  To convert v2 DSL code
    to v3 format, use the `./utils/bin/convert_v2_stencil.pl` utility.
    Conversion is recommended.
  - For both the compiler and kernel APIs, all uses of the term "grid" were
    changed to "var".  (Historically, early versions of YASK allowed only
    variables whose elements were points on the domain grid, so the terms
    were essentially interchangeable. Later, variables became more flexible.
    They could be defined with a subset of the domain dimensions, include
    non-domain or "miscellaneous" indices, or even be simple scalar values,
    so the term "grid" to describe any variable became inaccurate. This
    change addresses that contradiction.) Again, backward-compatibility
    features in the API should maintain functionality of v2 DSL and kernel
    code.
  - The default strings used in the kernel library and filenames to identify
    the targeted architecture were changed from Intel CPU codenames to
    [approximate] instruction-set architecture (ISA) names "avx512", "avx2",
    "avx", "knl", "knc", or "intel64". The YASK targets used in the YASK
    compiler were updated to be consistent with this list.
  - The "mid" (roughly, median) performance results are now the first
    ones printed by the `utils/bin/yask_log_to_csv.pl` script.
  - In general, any old DSL and kernel code or user-written output-parsing
    scripts that use any undocumented files, data, or types may have to be
    updated.

### Version 2
* Version 2.22.00 changed the heuristic to determine vector-folding sizes when some
sizes are specified. This did not affect the default folding sizes.
* Version 2.21.02 simplified the example 3-D stencils (`3axis`, `3plane`, etc.)
to calculate simple averages like those in the MiniGhost benchmark.
This reduced the number of floating-point operations but not the number of points read
for each stencil.
* Version 2.20.00 added checking of the step-dimension index value in the
`yk_grid::get_element()` and similar APIs.
Previously, invalid values silently "wrapped" around to valid values.
Now, by default, the step index must be valid when reading, and the valid step
indices are updated when writing.
The old behavior of silent index wrapping may be restored via `set_step_wrap(true)`.
The default for all `strict_indices` API parameters is now `true` to catch more
programming errors and
increase consistency of behavior between "set" and "get" APIs.
Also, the advanced `share_storage()` APIs have been replaced with `fuse_grids()`.
* Version 2.19.01 turned off multi-pass tuning by default. Enable with `-auto_tune_each_pass`.
* Version 2.18.03 allowed the default radius to be stencil-specific and changed the names
of example stencil "9axis" to "3axis_with_diags".
* Version 2.18.00 added the ability to specify the global-domain size, and it will calculate
the local-domain sizes from it.
There is no longer a default local-domain size.
Output changed terms "overall-problem" to "global-domain" and "rank-domain" to "local-domain".
* Version 2.17.00 determined the host architecture in `make` and `bin/yask.sh` and
number of MPI ranks in `bin/yask.sh`.
This changed the old behavior of `make` defaulting to `snb` architecture and
`bin/yask.sh` requiring `-arch` and `-ranks`.
Those options are still available to override the host-based default.
* Version 2.16.03 moved the position of the log-file name to the last column in the CSV
output of `utils/bin/yask_log_to_csv.pl`.
* Version 2.15.04 required a call to `yc_grid::set_dynamic_step_alloc(true)` to allow changing the
allocation in the step (time) dimension at run-time for grid variables created at YASK compile-time.
* Version 2.15.02 required all "misc" indices to be yask-compiler-time constants.
* Version 2.14.05 changed the meaning of temporal sizes so that 0 means never do temporal
blocking and 1 allows blocking within a single time-step for multi-pack solutions.
The default setting is 0, which keeps the old behavior.
* Version 2.13.06 changed the default behavior of the performance-test utility (`yask.sh`)
to run trials for a given amount of time instead of a given number of steps.
As of version 2.13.08, use the `-trial_time` option to specify the number of seconds to run.
To force a specific number of trials as in previous versions, use the `-trial_steps` option.
* Version 2.13.02 required some changes in perf statistics due to step (temporal) conditions.
Both text output and `yk_stats` APIs affected.
* Version 2.12.00 removed the long-deprecated `==` operator for asserting equality between
a grid point and an equation. Use `EQUALS` instead.
* Version 2.11.01 changed the plain-text format of some of the performance data in the
test-utility output.
Specifically, some leading spaces were added, SI multipliers for values < 1 were added,
and the phrase "time in" no longer appears before each time breakdown.
This may affect some user programs that parse the output to collect stats.
* Version 2.10.00 changed the location of temporary files created during the build process.
This will not affect most users, although you may need to manually remove old `src/compiler/gen`
and `src/kernel/gen` directories.
* Version 2.09.00 changed the location of stencils in the internal DSL from `.hpp` to `.cpp` files.
See the notes in https://github.com/intel/yask/releases/tag/v2.09.00 if you have any new
or modified code in `src/stencils`.
