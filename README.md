# YASK--Yet Another Stencil Kit

* New YASK users may want to start with the [YASK tutorial](http://intel.github.io/yask/YASK-tutorial.pdf).
* Users with existing YASK-based code may want to jump to the [backward-compatibility notices](#backward-compatibility-notices).

## Overview
YASK is a framework to rapidly create high-performance stencil code including optimizations and features such as
* Support for boundary layers and staggered-grid stencils.
* Vector-folding to increase data reuse via non-traditional data layout.
* Multi-level OpenMP parallelism to exploit multiple cores and threads.
* Scaling to multiple sockets and nodes via MPI with overlapped communication and compute.
* Spatial tiling with automatically-tuned block sizes.
* Temporal tiling in multiple dimensions to further increase cache locality.
* APIs for C++ and Python: [API documentation](http://intel.github.io/yask/api/html/index.html)

YASK contains a domain-specific compiler to convert stencil-equation specifications to SIMD-optimized code for Intel(R) Xeon Phi(TM) and Intel(R) Xeon(R) processors.

### Supported Platforms and Processors:
* 64-bit Linux.
* Intel(R) Xeon(R) processors supporting the AVX, AVX2, or CORE_AVX512 instruction sets.
* Intel(R) Xeon Phi(TM) x200-family processors supporting the MIC_AVX512 instruction set.
* Intel(R) Xeon Phi(TM) x100-family coprocessors supporting the Knights-Corner instruction set (no longer tested).

### Pre-requisites:
* Intel(R) Parallel Studio XE Cluster Edition for Linux
  for multi-socket and multi-node operation or
  Intel(R) Parallel Studio XE Composer Edition for C++ Linux
  for single-socket only
  (2018 or later; 2019.3 or later recommended).
     * There was an issue in Intel(R) MPI versions 2019.1 and 2019.2 that
       caused the application to crash when allocating very
       large shared-memory (shm) regions, so those
       versions are not recommended when using the `-use_shm` feature.
       This issue was resolved in MPI version 2019.3.
     * If you are using g++ version 8.x or later, Intel(R) C++ version 2019.x or later
       is required.
     * Building a YASK kernel with the Gnu C++ compiler is possible.
       Limited testing with g++ 8.2.0 shows the "iso3dfd" kernel
       runs about 30% slower compared to the same kernel built with
       the Intel C++ compiler.
       Older Gnu C++ compilers can produce kernels that run
       many times slower.
* Gnu C++ compiler, g++ (4.9.0 or later; 9.1.0 or later recommended).
  Even when using Intel compilers, they rely on functionality provided by a g++ installation.
* Linux libraries `librt` and `libnuma`.
* Perl (5.010 or later).
* Awk.
* Gnu make.
* Bash shell.
* Numactl utility.
* Optional utilities and their purposes:
    * The `indent` or `gindent` utility, used automatically during the build process
      to make the generated code easier for humans to read.
      You'll get a warning when running `make` if one of these doesn't exist.
      Everything will still work, but the generated code will be difficult to read.
      Reading the generated code is only necessary for debug or curiosity.
    * SWIG (3.0.12 or later; 4.0.0 or later recommended),
      http://www.swig.org, for creating the Python interface.
    * Python 2 (2.7.5 or later) or 3 (3.6.1 or later),
      https://www.python.org/downloads, for creating and using the Python interface.
    * Doxygen (1.8.11 or later),
      http://doxygen.org, for creating updated API documentation.
      If you're not changing the API documentation, you can view the existing documentation
      at the link at the top of this page.
    * Graphviz (2.30.1 or later),
      http://www.graphviz.org, for rendering stencil diagrams.
    * Intel(R) Software Development Emulator,
      https://software.intel.com/en-us/articles/intel-software-development-emulator,
      for functional testing if you don't have native support for any given instruction set.

## Backward-compatibility notices, including changes in default behavior
### Version 3

* Version 3.00.00 was a major release with a number of backward-compatibility notices:

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
This reduced the number of floating-point operations but not the number of points read for each stencil.
* Version 2.20.00 added checking of the step-dimension index value in the `yk_grid::get_element()` and similar APIs.
Previously, invalid values silently "wrapped" around to valid values.
Now, by default, the step index must be valid when reading, and the valid step indices are updated when writing.
The old behavior of silent index wrapping may be restored via `set_step_wrap(true)`.
The default for all `strict_indices` API parameters is now `true` to catch more programming errors and
increase consistency of behavior between "set" and "get" APIs.
Also, the advanced `share_storage()` APIs have been replaced with `fuse_grids()`.
* Version 2.19.01 turned off multi-pass tuning by default. Enable with `-auto_tune_each_pass`.
* Version 2.18.03 allowed the default radius to be stencil-specific and changed the names of example stencil "9axis" to "3axis_with_diags".
* Version 2.18.00 added the ability to specify the global-domain size, and it will calculate the local-domain sizes from it.
There is no longer a default local-domain size.
Output changed terms "overall-problem" to "global-domain" and "rank-domain" to "local-domain".
* Version 2.17.00 determined the host architecture in `make` and `bin/yask.sh` and number of MPI ranks in `bin/yask.sh`.
This changed the old behavior of `make` defaulting to `snb` architecture and `bin/yask.sh` requiring `-arch` and `-ranks`.
Those options are still available to override the host-based default.
* Version 2.16.03 moved the position of the log-file name to the last column in the CSV output of `utils/bin/yask_log_to_csv.pl`.
* Version 2.15.04 required a call to `yc_grid::set_dynamic_step_alloc(true)` to allow changing the
allocation in the step (time) dimension at run-time for grid variables created at YASK compile-time.
* Version 2.15.02 required all "misc" indices to be yask-compiler-time constants.
* Version 2.14.05 changed the meaning of temporal sizes so that 0 means never do temporal blocking and 1 allows blocking within a single time-step for multi-pack solutions. The default setting is 0, which keeps the old behavior.
* Version 2.13.06 changed the default behavior of the performance-test utility (`yask.sh`) to run trials for a given amount of time instead of a given number of steps. As of version 2.13.08, use the `-trial_time` option to specify the number of seconds to run. To force a specific number of trials as in previous versions, use the `-trial_steps` option.
* Version 2.13.02 required some changes in perf statistics due to step (temporal) conditions. Both text output and `yk_stats` APIs affected.
* Version 2.12.00 removed the long-deprecated `==` operator for asserting equality between a grid point and an equation. Use `EQUALS` instead.
* Version 2.11.01 changed the plain-text format of some of the performance data in the test-utility output. Specifically, some leading spaces were added, SI multipliers for values < 1 were added, and the phrase "time in" no longer appears before each time breakdown. This may affect some user programs that parse the output to collect stats.
* Version 2.10.00 changed the location of temporary files created during the build process. This will not affect most users, although you may need to manually remove old `src/compiler/gen` and `src/kernel/gen` directories.
* Version 2.09.00 changed the location of stencils in the internal DSL from `.hpp` to `.cpp` files. See the notes in https://github.com/intel/yask/releases/tag/v2.09.00 if you have any new or modified code in `src/stencils`.
