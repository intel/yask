##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2022, Intel Corporation
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
# Set YASK_OUTPUT_DIR to change where all output files go.

# Some of the make vars that are commonly set via the command-line
#   and passed to src/kernel/Makefile are listed here.
#   The 'stencil' and 'arch' vars are most important and should always be specified.
#
# stencil: sets stencil problem to be solved.
#   For a list of current stencils, run the following:
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
# fold: In which dimension(s) to vectorize.
# cluster: How many vectors to evaluate simultaneously.
#
# pfd_l1: L1 prefetch distance (0 => disabled).
# pfd_l2: L2 prefetch distance (0 => disabled).

# Common defaults.
offload			?=	0

# Common settings.
YASK_BASE	:=	$(abspath .)
include $(YASK_BASE)/src/common/common.mk

# This is mostly a wrapper for building several parts of YASK via src/*/Makefile.
# See those specific files for many more settings.
# Convention: when useful for distinction,
# - vars starting with 'YK_' apply to the YASK stencil kernel.
# - vars starting with 'YC_' apply to the YASK stencil compiler.

#YASK_MFLAGS	:=	--max-load 16
#YASK_MFLAGS	+=	--output-sync --output-sync=line
YK_MAKE		:=	$(MAKE) $(YASK_MFLAGS) -C src/kernel YASK_OUTPUT_DIR=$(YASK_OUT_BASE)
YC_MAKE		:=	$(MAKE) $(YASK_MFLAGS) -C src/compiler YASK_OUTPUT_DIR=$(YASK_OUT_BASE)

# Compiler and default flags--used only for targets in this Makefile.
# For compiler, use YC_CXX*.
# For kernel, use YK_CXX*.
CXX		:=	g++
CXXOPT		:=	-O2
CXXFLAGS 	:=	-g -std=c++11 -Wall $(CXXOPT)
CXXFLAGS	+=	$(addprefix -I,$(INC_DIR) $(COMM_DIR))
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

py-kernel-api:
	$(YK_MAKE) py-api

api:
	$(YC_MAKE) $@
	$(YK_MAKE) $@

# Remove old generated API documents and make new ones.
docs/api/html/index.html: include/*.hpp include/*/*.hpp docs/api/*.*
	doxygen -v
	find docs/api/html -type f | xargs -r rm
	cd docs/api; doxygen doxygen_config.txt

######## Misc targets

code-stats:
	$(YK_MAKE) $@

docs: api-docs

tags:
	rm -f TAGS ; find src include -name '*.[ch]pp' | xargs etags -C -a

# Remove intermediate files.
# Should not trigger remake of stencil compiler, so does not invoke clean in compiler dir.
# Make this target before rebuilding YASK with any new parameters.
clean:
	$(YK_MAKE) $@

# Remove executables, generated documentation, etc. (not logs).
# Use 'find *' instead of 'find .' to avoid searching in '.git'.
realclean: clean
	rm -rf $(LIB_OUT_DIR) $(BIN_OUT_DIR) $(BUILD_OUT_DIR)
	rm -fv TAGS '*~'
	- find src include utils -name '*~' -print -delete
	- find src -name '*.optrpt' -print -delete
	- find src -name __pycache__ -print -delete
	$(YC_MAKE) $@
	$(YK_MAKE) $@
	- find $(PY_OUT_DIR) -mindepth 1 '!' -name __init__.py -print -delete
	- rmdir -v --ignore-fail-on-non-empty $(PY_OUT_DIR)
	- rmdir -v --ignore-fail-on-non-empty $(YASK_OUT_BASE)

help:
	@ $(YC_MAKE) $@
	@ $(YK_MAKE) $@
	@ echo " "
	@ echo "'setenv CXX_PREFIX ccache' or 'export CXX_PREFIX=ccache' to use ccache."

#################################
########### Tests ###############
#################################
# TODO: convert all testing to a separate test framework.

# Test dirs & files.
TUPLE_TEST_EXEC :=	$(BIN_OUT_DIR)/yask_tuple_test.exe
COMBO_TEST_EXEC :=	$(BIN_OUT_DIR)/yask_combo_test.exe

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

# Run 8 out of 9 combos of (built-in, C++, Python)^2
# API tests. The 9th one is built-in with built-in,
# which is tested more extensively in the kernel tests.
# When the built-in stencil examples aren't being used,
# "stencil=api_test" in the commands below is simply used to
# create file names.
yc-combo-api-tests:
	$(MAKE) clean; $(MAKE) stencil=iso3dfd yc-and-cxx-yk-api-test
	$(MAKE) clean; $(MAKE) stencil=iso3dfd yc-and-py-yk-api-test

cxx-yc-combo-api-tests:
	$(MAKE) clean; $(MAKE) stencil=api_test cxx-yc-api-and-yk-test
	$(MAKE) clean; $(MAKE) stencil=api_test cxx-yc-api-and-cxx-yk-api-test
	$(MAKE) clean; $(MAKE) stencil=api_test cxx-yc-api-and-py-yk-api-test

py-yc-combo-api-tests:
	$(MAKE) clean; $(MAKE) stencil=api_test py-yc-api-and-yk-test
	$(MAKE) clean; $(MAKE) stencil=api_test py-yc-api-and-py-yk-api-test
	$(MAKE) clean; $(MAKE) stencil=api_test py-yc-api-and-cxx-yk-api-test

combo-api-tests:
	$(MAKE) yc-combo-api-tests
	if (( $(offload) == 0 )); then \
	  $(MAKE) cxx-yc-combo-api-tests && \
	  $(MAKE) py-yc-combo-api-tests; fi

# NB: set arch var if applicable.
# NB: save some test time by using YK_CXXOPT=-O2.

yc-and-yk-test:
	$(YK_MAKE) $@

$(TUPLE_TEST_EXEC): $(COMM_DIR)/tests/tuple_test.cpp $(COMM_DIR)/*.*pp
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $< $(COMM_DIR)/tuple.cpp $(COMM_DIR)/common_utils.cpp

$(COMBO_TEST_EXEC): $(COMM_DIR)/tests/combo_test.cpp $(COMM_DIR)/*.*pp
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $< $(COMM_DIR)/combo.cpp

tuple-test: $(TUPLE_TEST_EXEC)
	@echo '*** Running the C++ YASK tuple test...'
	$(RUN_PREFIX) $<

combo-test: $(COMBO_TEST_EXEC)
	@echo '*** Running the C++ YASK combo test...'
	$(RUN_PREFIX) $<

api-tests: compiler-api
	$(MAKE) combo-api-tests
	$(YK_MAKE) $@

unit-tests:
	$(MAKE) tuple-test
	$(MAKE) combo-test

all-tests: compiler-api unit-tests
	$(YK_MAKE) $@
	$(MAKE) combo-api-tests
	$(MAKE) clean
	@echo "All YASK tests have been run"

all:
	$(MAKE) realclean
	$(MAKE) tags
	$(MAKE) default
	$(MAKE) api-all
	$(MAKE) all-tests
