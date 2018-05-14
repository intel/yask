YASK--Yet Another Stencil Kernel: A framework to rapidly create high-performance HPC stencil code including optimizations and features such as
* Vector folding,
* Cache blocking,
* Multi-level OpenMP parallelism,
* Encapsulated memory layout,
* Temporal wave-front blocking,
* MPI halo exchange, and
* APIs for C++ and Python: [API documentation]{https://rawgit.com/intel/yask/api-docs/html/index.html}

YASK contains a domain-specific compiler to convert scalar C++ stencil code to SIMD-optimized code for Intel(R) Xeon Phi(TM) and Intel(R) Xeon(R) processors.

Supported Platforms and Processors:
* 64-bit Linux
* Intel(R) Xeon(R) processors supporting the AVX, AVX2, or CORE_AVX512 instruction sets.
* Intel(R) Xeon Phi(TM) processors supporting the MIC_AVX512 instruction set.
* Intel(R) Xeon Phi(TM) coprocessors supporting the Knights-Corner instruction set (no longer tested).

Pre-requisites:
* Intel(R) Parallel Studio XE Cluster Edition for Linux
  for multi-socket and multi-node operation or
  Intel(R) Parallel Studio XE Composer Edition for C++ Linux
  for single-socket only
  (2016 or later, 2018 update 2 recommended).
* Gnu C++ compiler, g++ (4.9.0 or later; 6.1.0 or later recommended).
* Linux libraries 'librt' and 'libnuma'.
* Perl (5.010 or later).
* Awk.
* Gnu make.
* Bash shell.
* The 'indent' or 'gindent' utility (optional: to make the generated code easier for humans to read).
* SWIG (3.0.12 or later),
  http://www.swig.org (optional: for creating the Python interface).
* Python 2 (2.7.5 or later) or 3 (3.6.1 or later),
  https://www.python.org/downloads (optional: for creating and using the Python interface).
* Doxygen (1.8.11 or later),
  http://doxygen.org (optional: for creating API documentation).
* Graphviz (2.30.1 or later),
  http://www.graphviz.org (optional: for rendering stencil diagrams).
* Intel(R) Software Development Emulator,
  https://software.intel.com/en-us/articles/intel-software-development-emulator
  (optional: for functional testing if you don't have native ISA support).
* Install all these pre-requisites and ensure that all
  tool and library paths are included in the proper environment variables.
* The YASK source code via 'git clone https://github.com/intel/yask'.

To continue with building and running, see YASK-intro.pdf in the docs directory.
