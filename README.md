YASK--Yet Another Stencil Kernel: A framework to facilitate exploration of the HPC stencil-performance design space, including any optimizations such as
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
* Intel(R) C++ compiler,
  https://software.intel.com/en-us/intel-parallel-studio-xe.
* Intel(R) Software Development Emulator,
  https://software.intel.com/en-us/articles/intel-software-development-emulator
  (optional: for functional testing if you don't have native ISA support).
* Intel(R) MPI Library, https://software.intel.com/en-us/intel-mpi-library,
  or equivalent (optional: for multi-socket and multi-node operation).
* Perl 5.010 or later.
* The 'indent' or 'gindent' utility (optional: to make the generated code easier for humans to read).
* Install all these pre-requisites and ensure that all
  tool and library paths are included in the proper environment variables.
* The YASK source code via 'git clone https://github.com/01org/yask'.

To continue with building and running, see YASK-intro.pdf in the docs directory.

Notice: If you are attempting to reproduce the example results from the "Intel Software Momentum Guide of July 26, 2016" marketing document, you will need the archived source from https://01.org/sites/default/files/downloads/yask/yask-20160526.tar_0.gz. This source package is not recommended for any other purpose because it does not contain many improvements and features in the current release.
