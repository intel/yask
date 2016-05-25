Yet Another Stencil Kernel: A framework to facilitate exploration of the HPC
stencil-performance design space, including

* Vector folding,
* Cache blocking,
* Memory layout,
* Loop construction,
* Temporal wave-front blocking,
* And many others.

YASK contains a specialized source-to-source translator to convert scalar
C++ stencil code to SIMD-optimized code for Intel(R) Xeon Phi(TM)
processors.

Supported Platforms
* 64-bit Linux
* Intel(R) Xeon(R) processor supporting AVX, AVX2, or CORE_AVX512 instruction sets
* Intel(R) Xeon Phi(TM) (co)processor supporting Knights-Corner or MIC_AVX512 instruction sets

Pre-requisites:
* Intel(R) C++ compiler,
  https://software.intel.com/en-us/intel-parallel-studio-xe.
* Intel(R) Software Development Emulator,
  https://software.intel.com/en-us/articles/intel-software-development-emulator
  (optional: for functional testing if you don't have native ISA support).
* Intel(R) MPI Library, https://software.intel.com/en-us/intel-mpi-library,
  or equivalent.
* Perl 5.010 or later.
* Install all these pre-requisites and ensure that all
  tool and library paths are included in the proper environment variables.
* Git-clone or download the YASK source code.

To continue with building and running, see YASK-intro.pdf in the docs directory.
