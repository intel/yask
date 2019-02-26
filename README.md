# YASK--Yet Another Stencil Kernel

* New YASK users may want to start with the [YASK tutorial](docs/YASK-tutorial.pdf).
* Existing YASK users may want to jump to the [backward-compatibility notices](#backward-compatibility-notices).

## Overview
YASK is a framework to rapidly create high-performance stencil code including optimizations and features such as
* Vector-folding to increase data reuse via non-traditional data layout,
* Multi-level OpenMP parallelism to exploit multiple cores and threads,
* Scaling to multiple sockets and nodes via MPI with overlapped communication and compute, and
* Spatial tiling with automatically-tuned block sizes,
* Temporal tiling to further increase cache locality,
* APIs for C++ and Python: [API documentation](https://rawgit.com/intel/yask/api-docs/html/index.html).

YASK contains a domain-specific compiler to convert scalar stencil code to SIMD-optimized code for Intel(R) Xeon Phi(TM) and Intel(R) Xeon(R) processors.

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
  (2018 or later; 2019 or later recommended and required when using g++ 8 or later).
  Building a YASK kernel with the Gnu compiler is possible, but only useful
  for functional testing. The performance
  of the kernel built from the Gnu compiler has been observed to be up to 7x lower
  than the same kernel built using the Intel compiler. 
* Gnu C++ compiler, g++ (4.9.0 or later; 8.2.0 or later recommended).
* Linux libraries `librt` and `libnuma`.
* Perl (5.010 or later).
* Awk.
* Gnu make.
* Bash shell.
* Numactl.
* Optional utilities and their purposes:
    * The `indent` or `gindent` utility, used automatically during the build process
      to make the generated code easier for humans to read.
      You'll get a warning when running `make` if one of these doesn't exist.
      Everything will still work, but the generated code will be difficult to read.
      Reading the generated code is only necessary for debug or curiosity.
    * SWIG (3.0.12 or later),
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

### Backward-compatibility notices:
* Version 2.18.03 allowed the default radius to be stencil-specific and changed the names of example stencils
"3axis" to "3line" and "9axis" to "9line".
* Version 2.18.00 added the ability to specify the global-domain size, and it will calculate the local-domain sizes from it.
There is no longer a default local-domain size.
Output changed terms "overall-problem" to "global-domain" and "rank-domain" to "local-domain".
* Version 2.17.00 determined the host architecture in `make` and `bin/yask.sh` and number of MPI ranks in `bin/yask.sh`.
This changed the old behavior of `make` defaulting to `snb` architecture and `bin/yask.sh` requiring `-arch` and `-ranks`.
Those options are still available to override the host-based default.
* Version 2.16.03 moved the position of the log-file name to the last column in the CSV output of `utils/bin/yask_log_to_csv.pl`.
* Version 2.15.04 required a call to `yc_grid::set_dynamic_step_alloc(true)` to allow changing the
allocation in the step (time) dimension for grid variables created at YASK compile-time.
* Version 2.15.02 required all "misc" indices to be yask-compiler-time constants.
* Version 2.14.05 changed the meaning of temporal sizes so that 0 means never do temporal blocking and 1 allows blocking within a single time-step for multi-pack solutions. The behavior of the default settings have not changed.
* Version 2.13.06 changed the default behavior of the performance-test utility (`yask.sh`) to run trials for a given amount of time instead of a given number of steps. As of version 2.13.08, use the `-trial_time` option to specify the number of seconds to run. To force a specific number of trials as in previous versions, use the `-trial_steps` option.
* Version 2.13.02 required some changes in perf statistics due to step (temporal) conditions. Both text output and `yk_stats` APIs affected.
* Version 2.12.00 removed the long-deprecated `==` operator for asserting equality between a grid point and an equation. Use `EQUALS` instead.
* Version 2.11.01 changed the plain-text format of some of the performance data in the test-utility output. Specifically, some leading spaces were added, SI multipliers for values < 1 were added, and the phrase "time in" no longer appears before each time breakdown. This may affect some user programs that parse the output to collect stats.
* Version 2.10.00 changed the location of temporary files created during the build process. This will not affect most users, although you may need to manually remove old `src/compiler/gen` and `src/kernel/gen` directories.
* Version 2.09.00 changed the location of stencils in the internal DSL from `.hpp` to `.cpp` files. See the notes in https://github.com/intel/yask/releases/tag/v2.09.00 if you have any new or modified code in `src/stencils`.
