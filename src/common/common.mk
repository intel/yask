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

# Common Makefile settings.
# YASK_BASE should be set before including this.

# YASK dirs.
LIB_DIR		:=	$(YASK_BASE)/lib
INC_DIR		:=	$(YASK_BASE)/include
BIN_DIR		:=	$(YASK_BASE)/bin
YASK_DIR	:=	$(YASK_BASE)/yask
SRC_DIR		:=	$(YASK_BASE)/src

# OS-specific
ifeq ($(shell uname -o),Cygwin)
  SO_SUFFIX	:=	.dll
  RUN_PREFIX	:=	env PATH="${PATH}:$(LIB_DIR):$(YASK_DIR)"
  PYTHON	:=	python3
else
  SO_SUFFIX	:=	.so
  RUN_PREFIX	:=
  PYTHON	:=	python
endif

# Common source.
COMM_DIR	:=	$(SRC_DIR)/common
COMM_SRC_NAMES	:=	output common_utils tuple
COMM_SRC_BASES	:=	$(addprefix $(COMM_DIR)/,$(COMM_SRC_NAMES))

# YASK stencil compiler.
# This is here because both the compiler and kernel
# Makefiles need to know about the compiler.
YC_BASE		:=	yask_compiler
YC_EXEC		:=	$(BIN_DIR)/$(YC_BASE).exe

# Tools.
SWIG		:=	swig
PERL		:=	perl

# Find include path needed for python interface.
# NB: constructing string inside print() to work for python 2 or 3.
PYINC		:= 	$(addprefix -I,$(shell $(PYTHON) -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc() + " " + distutils.sysconfig.get_python_inc(plat_specific=1))'))

RUN_PYTHON	:= 	$(RUN_PREFIX) env PYTHONPATH=$(LIB_DIR):$(YASK_DIR):$(PYTHONPATH) $(PYTHON)

