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
* Intel(R) Xeon Phi(TM) processor (for performance), or any Intel Architecture platform (for functional testing)

Pre-requisites:
* Intel(R) C++ compiler,
  https://software.intel.com/en-us/intel-parallel-studio-xe.
* Intel(R) Software Development Emulator (optional, for functional testing),
  https://software.intel.com/en-us/articles/intel-software-development-emulator.
* Perl 5.010 or later with the following libraries installed
  * String::Tokenizer,
    http://search.cpan.org/~stevan/String-Tokenizer-0.05/lib/String/Tokenizer.pm
  * Algorithm::Loops,
    http://search.cpan.org/~tyemq/Algorithm-Loops-1.031/lib/Algorithm/Loops.pm
* Install all these pre-requisites  and ensure that all
  tool and library paths are in the proper environment variables.
* Download and unzip the YASK source-code.

Example build commands:
* 'cd' to the directory containing the Makefile.
* Type one of the following commands, depending on your target architecture:
  * 'make arch=knc' for an Intel Xeon Phi coprocessor (MIC).
  * 'make arch=knl' for 'MIC-AVX512' code generation.
* For other build options, see the comments in the Makefile. Example options:
  * Stencil type.
  * Stencil order (size).
  * Vector folding.
  * Loop ordering.

Example run commands:
* Type  command to run with the default settings:
  * './stencil-run.sh -mic <n>' to run natively on Xeon Phi card number <n>.
  * 'sde -knl -- ./stencil-run.sh -arch knl' to run the MIC-AVX512 version on
    the emulator.
* For other run options, add '-help' to one of the above commands.

For more information, visit https://01.org/yask.
