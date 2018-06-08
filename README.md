*Notices:* 
* Version 2.10.00 changed the location of temporary files created during the build process. This will not affect most users, although you may need to manually remove old `src/compiler/gen` and `src/kernel/gen` directories.
* Version 2.09.00 changed the location of stencils in the internal DSL from `.hpp` to `.cpp` files. See the notes in https://github.com/intel/yask/releases/tag/v2.09.00 if you have any new or modified code in `src/stencils`.

## Overview
YASK--Yet Another Stencil Kernel: A framework to rapidly create high-performance HPC stencil code including optimizations and features such as
* Vector folding,
* Automatically-tuned cache blocking,
* Multi-level OpenMP parallelism,
* Encapsulated memory layout,
* Temporal wave-front blocking,
* MPI halo exchange, and
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
  (2016 or later, 2018 update 2 recommended).
  Building a YASK kernel with the Gnu compiler is possible, but only useful
  for functional testing. The performance
  of the kernel built from the Gnu compiler has been observed to be up to 7x lower
  than the same kernel built using the Intel compiler. 
* Gnu C++ compiler, g++ (4.9.0 or later; 6.1.0 or later recommended).
* Linux libraries 'librt' and 'libnuma'.
* Perl (5.010 or later).
* Awk.
* Gnu make.
* Bash shell.
* Optional utilities and their purposes:
    * The `indent` or `gindent` utility, used automatically during the build process
      to make the generated code easier for humans to read.
      You'll get a warning when running `make` if one of these doesn't exist.
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

To continue with building and running, see YASK-intro.pdf in the docs directory.
