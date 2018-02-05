##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2018, Intel Corporation
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## * The above copyright notice and this permission notice shall be included in
##   all copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
##############################################################################

# Makefile for YASK.
# Type 'make help' for usage.

# Some of the make vars that are commonly set via the command-line
#   and passed to src/kernel/Makefile are listed here.
#   The 'stencil' and 'arch' vars are most important and should always be specified.
#
# stencil: sets stencil problem to be solved.
#   For a list of current stencils, see src/kernel/Makefile or run the following:
#   % make compiler
#   % bin/yask_compiler.exe -h
#   You can also create your own stencil; see the documentation.
#
# arch: sets target architecture for best performance.
#   For a list of archs, see src/kernel/Makefile.
#
# mpi: 0, 1: whether to use MPI. 
#
# radius: sets size of certain stencils.
#
# real_bytes: FP precision: 4=float, 8=double.
#
# fold: How to fold vectors (x*y*z).
#   Vectorization in dimensions perpendicular to the inner loop
#   (defined by SUB_BLOCK_LOOP_INNER_VARS below) often works well.
# fold_4byte: How to fold vectors when real_bytes=4.
# fold_8byte: How to fold vectors when real_bytes=8.
#
# cluster: How many folded vectors to evaluate simultaneously.
#
# pfd_l1: L1 prefetch distance (0 => disabled).
# pfd_l2: L2 prefetch distance (0 => disabled).
#
# omp_region_schedule: OMP schedule policy for region loop.
# omp_block_schedule: OMP schedule policy for nested OpenMP block loop.
# omp_halo_schedule: OMP schedule policy for OpenMP halo loop.
#
# def_block_threads: Number of threads to use in nested OpenMP block loop by default.
# def_thread_divisor: Divide number of OpenMP threads by this factor by default.
# def_*_args: Default cmd-line args for specific settings.
# more_def_args: Additional default cmd-line args.
#
# max_dims: max number of dimensions allowed in a grid.
#
# eqs: comma-separated name=substr pairs used to manually group
#   grid-update equations into sets.

# This is mostly a wrapper for building several parts of YASK via src/*/Makefile.
# See those specific files for many more settings.
# Convention: when useful for distinction,
# - vars starting with 'YK_' apply to the YASK stencil kernel.
# - vars starting with 'YC_' apply to the YASK stencil compiler.

YK_MAKE	:=	$(MAKE) -C src/kernel
YC_MAKE	:=	$(MAKE) -C src/compiler

# YASK dirs.
YASK_BASE	:=	$(shell pwd)
LIB_DIR		:=	$(YASK_BASE)/lib
INC_DIR		:=	$(YASK_BASE)/include
BIN_DIR		:=	$(YASK_BASE)/bin

# OS-specific
ifeq ($(shell uname -o),Cygwin)
  SO_SUFFIX	:=	.dll
  RUN_PREFIX	:=	env PATH="${PATH}:$(LIB_DIR)"
  PYTHON		:= python3
else
  SO_SUFFIX	:=	.so
  RUN_PREFIX	:=
  PYTHON		:=	python
  CXX_PREFIX	:=	time
endif

# Misc dirs & files.
COMM_DIR	:=	src/common
TUPLE_TEST_EXEC :=	$(BIN_DIR)/yask_tuple_test.exe

# Compiler and default flags--used only for targets in this Makefile.
# For compiler, use YC_CXX*.
# For kernel, use YK_CXX*.
CXX		:=	g++
CXXFLAGS 	:=	-g -std=c++11 -Wall -O2
CXXFLAGS	+=	$(addprefix -I,$(COMM_DIR))
CXXFLAGS	+=	-fopenmp

######## Primary targets & rules
# NB: must set stencil and arch to generate the desired kernel.

default:
	$(YC_MAKE) $@
	$(YK_MAKE) $@

compiler:
	$(YC_MAKE) $@

kernel:
	$(YK_MAKE) $@

######## API targets
# NB: must set stencil and arch to generate the desired kernel API.

# API docs & libs.
api-all: api-docs api

# Format API documents.
api-docs: docs/api/html/index.html
	@ echo 'Open the following file in a browser to view the API docs.'
	@ ls -l $^

# Build C++ and Python API libs.
compiler-api:
	$(YC_MAKE) api

kernel-api:
	$(YK_MAKE) api

api:
	$(YC_MAKE) $@
	$(YK_MAKE) $@

# Format API documents.
docs/api/html/index.html: include/*.hpp docs/api/*.*
	doxygen -v
	cd docs/api; doxygen doxygen_config.txt

#### API tests.

# The tests listed here are designed to test various combinations of the
# compiler and kernel options.  For tests focused on the compiler or kernel,
# see their corresponding Makefiles.

# Run C++ compiler API test, then run YASK kernel using its output.
cxx-yc-api-and-yk-test:
	$(YK_MAKE) cxx-yc-api-test
	$(YK_MAKE) yk-test-no-yc

# Run Python compiler API test, then run YASK kernel using its output.
py-yc-api-and-yk-test:
	$(YK_MAKE) py-yc-api-test
	$(YK_MAKE) yk-test-no-yc

# Run built-in compiler, then run C++ kernel API test.
yc-and-cxx-yk-api-test:
	$(YK_MAKE) code-file
	$(YK_MAKE) cxx-yk-api-test

# Run built-in compiler, then run python kernel API test.
yc-and-py-yk-api-test:
	$(YK_MAKE) code-file
	$(YK_MAKE) py-yk-api-test

# Run C++ compiler API test, then run C++ kernel API test.
cxx-yc-api-and-cxx-yk-api-test:
	$(YK_MAKE) cxx-yc-api-test
	$(YK_MAKE) cxx-yk-api-test

# Run python compiler API test, then run python kernel API test.
py-yc-api-and-py-yk-api-test:
	$(YK_MAKE) py-yc-api-test
	$(YK_MAKE) py-yk-api-test

# Run C++ compiler API test, then run python kernel API test.
cxx-yc-api-and-py-yk-api-test:
	$(YK_MAKE) cxx-yc-api-test
	$(YK_MAKE) py-yk-api-test

# Run python compiler API test, then run C++ kernel API test.
py-yc-api-and-cxx-yk-api-test:
	$(YK_MAKE) py-yc-api-test
	$(YK_MAKE) cxx-yk-api-test

# Run C++ compiler API test with exception, then run C++ kernel API test with exception.
cxx-yc-api-and-cxx-yk-api-test-with-exception:
	$(YK_MAKE) cxx-yc-api-test-with-exception
	$(YK_MAKE) cxx-yk-api-test-with-exception

# Run python compiler API test with exception, then run python kernel API test with exception.
py-yc-api-and-py-yk-api-test-with-exception:
	$(YK_MAKE) py-yc-api-test-with-exception
	$(YK_MAKE) py-yk-api-test-with-exception

# Run C++ compiler API test with exception, then run python kernel API test with exception.
cxx-yc-api-and-py-yk-api-test-with-exception:
	$(YK_MAKE) cxx-yc-api-test-with-exception
	$(YK_MAKE) py-yk-api-test-with-exception

# Run python compiler API test with exception, then run C++ kernel API test with exception.
py-yc-api-and-cxx-yk-api-test-with-exception:
	$(YK_MAKE) py-yc-api-test-with-exception
	$(YK_MAKE) cxx-yk-api-test-with-exception

api-tests:
	$(MAKE) yc-and-cxx-yk-api-test
	$(MAKE) yc-and-py-yk-api-test
	$(MAKE) stencil=test cxx-yc-api-and-yk-test
	$(MAKE) stencil=test py-yc-api-and-yk-test
	$(MAKE) stencil=test cxx-yc-api-and-cxx-yk-api-test
	$(MAKE) stencil=test py-yc-api-and-py-yk-api-test
	$(MAKE) stencil=test cxx-yc-api-and-py-yk-api-test
	$(MAKE) stencil=test py-yc-api-and-cxx-yk-api-test
	$(MAKE) stencil=test cxx-yc-api-and-cxx-yk-api-test-with-exception
	$(MAKE) stencil=test py-yc-api-and-py-yk-api-test-with-exception
	$(MAKE) stencil=test cxx-yc-api-and-py-yk-api-test-with-exception
	$(MAKE) stencil=test py-yc-api-and-cxx-yk-api-test-with-exception

######## Misc targets

# NB: set arch var if applicable.
# NB: save some test time by using YK_CXXOPT=-O2.

yc-and-yk-test:
	$(YK_MAKE) $@

code-stats:
	$(YK_MAKE) $@

$(TUPLE_TEST_EXEC): src/common/tests/tuple_test.cpp src/common/tuple.hpp
	$(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $<

tuple-test: $(TUPLE_TEST_EXEC)
	@echo '*** Running the C++ YASK tuple test...'
	$(RUN_PREFIX) $<

all-tests: compiler
	$(MAKE) tuple-test
	$(YK_MAKE) $@
	$(MAKE) api-tests

docs: api-docs

all:
	$(MAKE) realclean
	$(MAKE) tags
	$(MAKE) default
	$(MAKE) all-tests
	$(MAKE) clean
	$(MAKE) default
	$(MAKE) api-all

docs: api-docs

tags:
	rm -f TAGS ; find src include -name '*.[ch]pp' | xargs etags -C -a

# Remove intermediate files.
# Should not trigger remake of stencil compiler, so does not invoke clean in compiler dir.
# Make this target before rebuilding YASK with any new parameters.
clean:
	$(YK_MAKE) $@

# Remove files from old versions.
# (This will eventually be removed.)
clean-old:
	rm -fv stencil*.exe stencil-tuner-summary.csh stencil-tuner.pl gen-layouts.pl gen-loops.pl get-loop-stats.pl
	rm -fv src/foldBuilder/*pp

# Remove executables, documentation, etc. (not logs).
realclean: clean-old
	rm -fv TAGS '*~'
	rm -fr docs/api/html
	rm -fr docs/api/latex
	rm -rf $(BIN_DIR)/*.exe $(LIB_DIR)/*$(SO_SUFFIX)
	- find * -name '*~' -print -delete
	- find * -name '*.optrpt' -print -delete
	- find * -name __pycache__ -print -delete
	- find yask -mindepth 1 '!' -name __init__.py -print -delete
	$(YC_MAKE) $@
	$(YK_MAKE) $@

help:
	@ $(YC_MAKE) $@
	@ $(YK_MAKE) $@
