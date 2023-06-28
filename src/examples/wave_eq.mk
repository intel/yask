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

# Makefile for the YASK example application 'wave_eq'.
stencil		?=	wave2d
real_bytes	?=	8

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# Binary.
WAVE_EQ_EXE	:=	$(BIN_OUT_DIR)/wave_eq.exe

### Make targets ###

default: $(WAVE_EQ_EXE)

$(WAVE_EQ_EXE): wave_eq_main.cpp $(YK_LIB)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) $(YK_LIBS) -o $@
	@ls -l $@

$(YK_LIB): FORCE
	$(MAKE) -C $(YASK_BASE) stencil=$(stencil) real_bytes=$(real_bytes)

FORCE:

