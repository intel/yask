YASK--Yet Another Stencil Kernel: A framework to facilitate exploration of the HPC stencil-performance design space, including optimizations such as
* Vector folding,
* Cache blocking,
* Memory layout,
* Loop construction,
* Temporal wave-front blocking, and
* MPI halo exchange.

YASK contains a specialized source-to-source translator to convert scalar C++ stencil code to SIMD-optimized code for Intel(R) Xeon Phi(TM) and Intel(R) Xeon(R) processors.

Supported Platforms
* 64-bit Linux
* Intel(R) Xeon Phi(TM) processor supporting the MIC_AVX512 instruction set.
* Intel(R) Xeon(R) processor supporting the AVX, AVX2, or CORE_AVX512 instruction sets
* Intel(R) Xeon Phi(TM) coprocessor supporting the Knights-Corner instruction set.

Pre-requisites:
* Intel(R) C++ compiler (17.0.2 or later recommended),
  https://software.intel.com/en-us/intel-parallel-studio-xe.
* Gnu C++ compiler, g++ (4.9.0 or later; 6.1.0 or later recommended).
* Intel(R) Software Development Emulator,
  https://software.intel.com/en-us/articles/intel-software-development-emulator
  (optional: for functional testing if you don't have native ISA support).
* Intel(R) MPI Library, https://software.intel.com/en-us/intel-mpi-library,
  or equivalent (optional: for multi-core and multi-node operation).
* Perl (5.010 or later).
* Awk.
* Gnu make.
* The 'indent' or 'gindent' utility (optional: to make the generated code easier for humans to read).
* SWIG (3.0.12 or later),
  http://www.swig.org (optional: for creating the Python interface).
* Python 2 (2.7.5 or later) or 3 (3.6.1 or later),
  https://www.python.org/downloads (optional: for creating and using the Python interface).
* Doxygen (1.8.11 or later),
  http://doxygen.org (optional: for creating API documentation).
* Graphviz (2.30.1 or later),
  http://www.graphviz.org (optional: for rendering stencil diagrams).
* Install all these pre-requisites and ensure that all
  tool and library paths are included in the proper environment variables.
* The YASK source code via 'git clone https://github.com/intel/yask'.

To continue with building and running, see YASK-intro.pdf in the docs directory.
