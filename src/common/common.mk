##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2020, Intel Corporation
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

# OS-specific
ifeq ($(shell uname -o),Cygwin)
  SO_SUFFIX	:=	.dll
  RUN_PREFIX	:=	env PATH="${PATH}:$(LIB_DIR):$(LIB_OUT_DIR):$(YASK_DIR):$(PY_OUT_DIR)"
  PYTHON	:=	python3
else
  SO_SUFFIX	:=	.so
  RUN_PREFIX	:=	env I_MPI_DEBUG=+5 I_MPI_PRINT_VERSION=1 OMP_DISPLAY_ENV=VERBOSE KMP_VERSION=1
  PYTHON	:=	python
endif
SHELL		:=	/bin/bash

# Common source.
COMM_DIR	:=	$(SRC_DIR)/common
COMM_SRC_NAMES	:=	output common_utils tuple combo fd_coeff fd_coeff2
COEFF_DIR	:=	$(SRC_DIR)/contrib/coefficients

# Globs and flags.
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
BASH		:=	bash

# Find include path needed for python interface.
# NB: constructing string inside print() to work for python 2 or 3.
PYINC		:= 	$(addprefix -I,$(shell $(PYTHON) -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc() + " " + distutils.sysconfig.get_python_inc(plat_specific=1))'))

RUN_PYTHON	:= 	$(RUN_PREFIX) \
	env PYTHONPATH=$(LIB_DIR):$(LIB_OUT_DIR):$(YASK_DIR):$(PY_OUT_DIR):$(PYTHONPATH) $(PYTHON)

# Function to check for pre-defined compiler macro.
# Invokes compiler using 1st arg.
# Returns '1' if 2nd arg is defined, '0' if not.
# Ex: "ifeq ($(call MACRO_DEF,$(CXX),__clang__),1)"...
MACRO_DEF	=	$(shell $(1) -x c++ /dev/null -dM -E | grep -m 1 -c $(2))

# Options to avoid warnings when compiling SWIG-generated code w/gcc.
SWIG_GCCFLAGS	:=	-Wno-class-memaccess -Wno-stringop-overflow -Wno-stringop-truncation

# Define deprecated macro used by SWIG.
DBL_EPSILON_CXXFLAG	:=	-DDBL_EPSILON=2.2204460492503131e-16
