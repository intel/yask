Yet Another Stencil Kernel: A framework to facilitate exploration of the HPC
stencil-performance design space, including

* Vector folding,
* Cache blocking,
* Memory layout,
* Loop construction,
* And many others.

YASK contains a specialized source-to-source translator to convert scalar
C++ stencil code to SIMD-optimized code for Intel(R) Xeon Phi(TM)
processors.

Supported Platform
* 64-bit Linux
* Intel(R) Xeon Phi(TM) coprocessor (for performance), or any Intel Architecture platform (for functional testing)

Pre-requisites:
* Intel(R) C++ compiler,
  https://software.intel.com/en-us/intel-parallel-studio-xe.
* Intel(R) Software Development Emulator (optional, for functional testing),
  https://software.intel.com/en-us/articles/intel-software-development-emulator.
* Perl 5.010 or later with the following modules installed
  * String::Tokenizer,
    http://search.cpan.org/~stevan/String-Tokenizer-0.05/lib/String/Tokenizer.pm
  * Algorithm::Loops,
    http://search.cpan.org/~tyemq/Algorithm-Loops-1.031/lib/Algorithm/Loops.pm
* Install all these pre-requisites and ensure that all
  tool and library paths are included in the proper environment variables.
* Git-clone or download the YASK source code.

To continue with building and running, see YASK-intro.pdf in the docs directory.
