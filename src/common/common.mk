##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2023, Intel Corporation
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

# Common Makefile settings.
# YASK_BASE should be set before including this.
# YASK_BASE and all the dirs based on it should be set with full (not relative) paths.

# Name strings.
TIMESTAMP	:=	$(shell date '+%Y-%m-%d')
HOSTNAME	:= 	$(shell hostname)

# Vars for special strings.
empty		:=
space		:=	$(empty) $(empty)
comma		:=	,
colon		:=	:

# Set YASK_OUTPUT_DIR to change where all output files go.
YASK_OUTPUT_DIR	?=	$(YASK_BASE)

# Top-level input dirs.
INC_DIR		:=	$(YASK_BASE)/include
YASK_DIR	:=	$(YASK_BASE)/yask
SRC_DIR		:=	$(YASK_BASE)/src
UTILS_DIR	:=	$(YASK_BASE)/utils
UTILS_BIN_DIR	:=	$(UTILS_DIR)/bin
UTILS_LIB_DIR	:=	$(UTILS_DIR)/lib

# Top-level output dirs.
YASK_OUT_BASE	:=	$(abspath $(YASK_OUTPUT_DIR))
LIB_OUT_DIR	:=	$(YASK_OUT_BASE)/lib
BIN_OUT_DIR	:=	$(YASK_OUT_BASE)/bin
BUILD_OUT_DIR	:=	$(YASK_OUT_BASE)/build
PY_OUT_DIR	:=	$(YASK_OUT_BASE)/yask
TEST_LOG_OUT_DIR :=	$(YASK_OUT_BASE)/logs/tests.$(HOSTNAME).$(TIMESTAMP)

# Common source.
COMM_DIR	:=	$(SRC_DIR)/common
COMM_SRC_NAMES	:=	output common_utils tuple combo fd_coeff fd_coeff2
COEFF_DIR	:=	$(SRC_DIR)/contrib/coefficients
INC_DIRS	:=	$(INC_DIR) $(INC_DIR)/aux
INC_GLOB	:=	$(wildcard $(addsuffix /*.hpp,$(INC_DIRS)))
INC_CXXFLAGS	:=	$(addprefix -I,$(INC_DIRS))

# YASK stencil compiler.
# This is here because both the compiler and kernel
# Makefiles need to know about the compiler.
YC_BASE		:=	yask_compiler
YC_EXEC		:=	$(BIN_OUT_DIR)/$(YC_BASE).exe
YC_SRC_DIR	:=	$(SRC_DIR)/compiler

# Tools.
SWIG		:=	swig
PERL		:=	perl
MKDIR		:=	mkdir -p -v
BASH		:=	/bin/bash
INDENT		:=	$(UTILS_BIN_DIR)/yask_indent.sh
PYTHON  	:=	python3
SHELL		:=	/bin/bash
SO_SUFFIX	:=	.so
RUN_PREFIX	:=	env I_MPI_DEBUG=+5 I_MPI_PRINT_VERSION=1 OMP_DISPLAY_ENV=VERBOSE KMP_VERSION=1
RUN_PYTHON	:= 	$(RUN_PREFIX) \
	env PYTHONPATH=$(LIB_DIR):$(LIB_OUT_DIR):$(PY_OUT_DIR):$(YASK_DIR):$(PYTHONPATH) $(PYTHON)

# Find include path needed for python interface.
# NB: constructing string inside print() to work for python 2 or 3.
PYINC		:= 	$(addprefix -I,$(shell $(PYTHON) -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc() + " " + distutils.sysconfig.get_python_inc(plat_specific=1))'))

# Function to check for pre-defined compiler macro.
# Invokes compiler using 1st arg.
# Returns '1' if 2nd arg is defined, '0' if not.
# Ex: "ifeq ($(call MACRO_DEF,$(CXX),__clang__),1)"...
MACRO_DEF	=	$(shell $(1) -x c++ /dev/null -dM -E | grep -m 1 -c $(2))

# Function to run a command serially, even with parallel build.
SERIALIZE	= 	exec {fd}>/tmp/$$USER.YASK.build-lock; \
			flock -x $$fd; \
			$(1)

# Function to create a directory.
# Tries to avoid the possible race condition when calling mkdir in parallel.
# 1st arg is dir name.
# Ex: "$(call MK_DIR,path)"
MK_DIR		=	@ if [ \! -d $(1) ]; then \
			  $(call SERIALIZE,$(MKDIR) $(1)); fi

# Script to remove unsupported function in python 3.8+.
SWIG_PATCH	:= perl -i -n -e 'print unless /_PyObject_GC_UNTRACK/'

# Options for compiling SWIG-generated code w/gcc.
SWIG_GCCFLAGS	:= -DYASK_DEPRECATED=''

# Define deprecated macro used by SWIG.
DBL_EPSILON_CXXFLAG	:=	-DDBL_EPSILON=2.2204460492503131e-16

# Determine default architecture by running kernel script w/special knob.
# (Do not assume 'yask.sh' has been installed in $(BIN_OUT_DIR) yet.)
arch			?=	$(shell $(BASH) $(SRC_DIR)/kernel/yask.sh -show_arch)

# Set 'TARGET' from 'arch', converting codenames and other aliases to ISA names.
# 'TARGET' is the canonical target name.
# The possible values must agree with those in the APIs and YASK compiler.
ifneq ($(filter $(arch),avx snb ivb),)
  TARGET		:=	avx
else ifneq ($(filter $(arch),avx2 hsw bdw),)
  TARGET		:=	avx2
else ifneq ($(filter $(arch),avx512-ymm avx512lo),)
  TARGET		:=	avx512-ymm
else ifneq ($(filter $(arch),avx512 avx512-zmm avx512hi avx512f skx skl clx icx spr),)
  TARGET		:=	avx512
else ifneq ($(filter $(arch),knl),)
  TARGET		:=	knl
else ifneq ($(filter $(arch),intel64 cpp),)
  TARGET		:=	intel64
else
  $(error Target not recognized; use arch=avx512, avx512-ymm, avx2, avx, knl, or intel64)
endif

# Set 'offload=1' to build device-offload (e.g., GPU) library.
# Set 'offload_usm=1' to build with unified shared mem model.
# Set 'offload_arch' to the target offload architecture.
offload			?=	0
offload_usm		?=	0
ifeq ($(offload_usm),1)
 offload		:=	1
endif
ifeq ($(offload),1)
 offload_arch		?=	spir64
endif

# Set 'stencil' to the name of the YASK solution to build.
stencil			?=	iso3dfd

# Set 'real_bytes' to number of bytes in a float (4 or 8).
real_bytes		?=	4

# Set 'mpi=0' to build without MPI support.
mpi			?=	1

# Set 'omp=0' to build without OpenMP support.
omp			?=	1

# Main vars for naming the libraries and executables.
# Set the following vars on 'make' cmd-line for corresponding effects:
# - YK_STENCIL and/or YK_ARCH to name libraries and executables differently.
YK_STENCIL	?=	$(stencil)$(YK_STENCIL_SUFFIX)
ifeq ($(offload),1)
  YK_ARCH	:=	$(arch).offload-$(offload_arch)
else
  YK_ARCH	:=	$(arch)
endif
YK_TAG  	:=	$(YK_STENCIL).$(YK_ARCH)

# Kernel lib file names.
YK_BASE		:=	yask_kernel
YK_EXT_BASE	:=	$(YK_BASE).$(YK_TAG)
YK_LIB		:=	$(LIB_OUT_DIR)/lib$(YK_EXT_BASE)$(SO_SUFFIX)

# Compiler for building kernel lib and apps.
CXX		:=	icpx
YK_CXX		:=	$(CXX)
MPI_CXX 	:=	mpiicpc
ifeq ($(mpi),1)
 YK_CXXCMD	:=	$(MPI_CXX) -cxx=$(YK_CXX)
else
 YK_CXXCMD	:=	$(YK_CXX)
endif
ifeq ($(offload),1)
 CXX_PREFIX	:=
endif

# Base compiler flags for building kernel lib and apps.
ifeq ($(offload),0)
 YK_CXXDBG	:=	-g
endif
YK_CXXOPT	:=	-O3
YK_CXXWARN	:=	-Wall
YK_CXXFLAGS	:=	-std=c++17 $(YK_CXXDBG) $(YK_CXXOPT) $(YK_CXXWARN) -I$(INC_DIR) $(EXTRA_YK_CXXFLAGS)
ifeq ($(mpi),1)
 YK_CXXFLAGS	+=	-DUSE_MPI
endif

# Linker flags.
YK_LIBS		:=	-lrt
YK_LFLAGS	:=	-Wl,-rpath=$(LIB_OUT_DIR) -L$(LIB_OUT_DIR) -l$(YK_EXT_BASE)

# Default number of ranks for running MPI tests.
# 4 tests in-plane diagonal exchanges for 2D and 3D tests.
# 8 tests all exchanges for 3D tests.
ifneq ($(mpi),1)
ranks	:=	1
else
ranks	:=	4
endif
