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

# Makefile for the YASK stencil compiler.
# Type 'make help' for usage.

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# Vars starting with 'YC_' apply to the stencil compiler.

# Compiler dirs.
YC_SWIG_DIR	:=	$(abspath ./swig)
YC_LIB_SRC_DIR	:=	$(abspath ./lib)
YC_TEST_SRC_DIR	:=	$(abspath ./tests)
YC_OUT_BASE	:=	$(BUILD_OUT_DIR)/compiler
YC_SWIG_OUT_DIR	:=	$(YC_OUT_BASE)/swig
YC_OBJ_DIR	:=	$(YC_OUT_BASE)/obj
YC_STENCIL_DIR	:= 	$(SRC_DIR)/stencils

# Compiler files.
YC_MODULE	:=	$(YC_BASE)
YC_LIB		:=	$(LIB_OUT_DIR)/lib$(YC_BASE)$(SO_SUFFIX)
YC_PY_LIB	:=	$(YASK_OUT_DIR)/_$(YC_MODULE)$(SO_SUFFIX)
YC_PY_MOD	:=	$(YASK_OUT_DIR)/$(YC_MODULE).py
YC_TEST_EXEC	:=	$(BIN_OUT_DIR)/$(YC_BASE)_api_test.exe
YC_TEST_EXEC_WITH_EXCEPTION	:=	$(BIN_OUT_DIR)/$(YC_BASE)_api_exception_test.exe
YC_SRC_NAMES	:=	Expr ExprUtils Grid Eqs Print Vec Cpp CppIntrin YaskKernel Soln
YC_STENCIL_NAMES:=	$(notdir $(patsubst %.cpp,%,$(wildcard $(YC_STENCIL_DIR)/*.cpp)))
YC_OBJS		:=	$(addprefix $(YC_OBJ_DIR)/,$(addsuffix .o,$(YC_SRC_NAMES) $(COMM_SRC_NAMES)))
YC_STENCIL_OBJS	:=	$(addprefix $(YC_OBJ_DIR)/,$(addsuffix .o,$(YC_STENCIL_NAMES)))
YC_INC_DIRS	:=	$(INC_DIR) $(YC_LIB_SRC_DIR) $(COMM_DIR)
YC_INC_GLOB	:=	$(wildcard $(addsuffix /*.hpp,$(YC_INC_DIRS)))
YC_STENCIL_INC_GLOB :=	$(wildcard $(YC_STENCIL_DIR)/*.hpp $(YC_STENCIL_DIR)/*/*.hpp)

# Compiler and default flags.
YC_CXX    	?=	g++  # usually faster than icpc for building the compiler.
YC_CXXOPT	:= 	-O2
YC_CXXFLAGS 	:=	-g -std=c++11 -Wall $(YC_CXXOPT)
YC_CXXFLAGS	+=	$(addprefix -I,$(YC_INC_DIRS) $(YC_STENCIL_DIR))

# Add user-defined flags.
YC_CXXFLAGS	+=	$(EXTRA_YC_CXXFLAGS)

# Linker.
YC_LD		:=	$(YC_CXX)
YC_LFLAGS	:=	-lrt -Wl,-rpath=$(LIB_OUT_DIR) -L$(LIB_OUT_DIR) -l$(YC_BASE)

# Compile rules.
# For example stencils,
# - Build at O0 to avoid C++ compiler wasting time optimizing them.
# - Set macro to use internal DSL instead of conflicting API operators.

$(YC_OBJ_DIR)/%.o: $(YC_STENCIL_DIR)/%.cpp $(YC_INC_GLOB) $(YC_STENCIL_INC_GLOB)
	$(MKDIR) $(YC_OBJ_DIR)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -DUSE_INTERNAL_DSL -O0 -c -o $@ $<
	@ls -l $@

$(YC_OBJ_DIR)/%.o: $(YC_LIB_SRC_DIR)/%.cpp $(YC_INC_GLOB)
	$(MKDIR) $(YC_OBJ_DIR)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -fPIC -c -o $@ $<
	@ls -l $@

$(YC_OBJ_DIR)/%.o: $(COMM_DIR)/%.cpp $(YC_INC_GLOB)
	$(MKDIR) $(YC_OBJ_DIR)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -fPIC -c -o $@ $<
	@ls -l $@

######## Primary targets.

default: compiler

# Compile the stencil compiler.
compiler: $(YC_EXEC)
	@echo "Binary" $(YC_EXEC) "has been built."
	$(MAKE) old-code-check

$(YC_LIB): $(YC_OBJS)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -shared -o $@ $^
	@ls -l $@

$(YC_EXEC): main.cpp $(YC_STENCIL_OBJS) $(YC_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YC_LD) $(YC_CXXFLAGS) $^ $(YC_LFLAGS) -o $@
	@ls -l $@
	$(MAKE) echo-settings

######## API targets

# Build C++ and Python compiler API libs.
# (Makes sure the compiler exists also, to avoid problems
# when trying to build the kernel w/o a compiler binary.)
api: compiler $(YC_PY_LIB)

# Build python compiler API lib.
# Also builds $(YC_PY_MOD)
$(YC_SWIG_OUT_DIR)/yask_compiler_api_wrap.cpp: $(YC_SWIG_DIR)/yask*.i $(INC_DIR)/*.hpp
	$(SWIG) -version
	$(MKDIR) $(YC_SWIG_OUT_DIR) $(YASK_OUT_DIR)
	$(SWIG) -v -DYC_MODULE=$(YC_MODULE) -cppext cpp \
	  -I$(INC_DIR) -I$(COMM_DIR) -I$(COMM_DIR)/swig \
	  -c++ -python -o $@ -outdir $(YASK_OUT_DIR) -builtin $<

# Turn off asserts to work around known SWIG issue:
# https://github.com/swig/swig/issues/773
$(YC_OBJ_DIR)/yask_compiler_api_wrap.o: $(YC_SWIG_OUT_DIR)/yask_compiler_api_wrap.cpp
	$(MKDIR) $(YC_OBJ_DIR)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -DNDEBUG $(PYINC) -fPIC -c -o $@ $<
	@ls -l $@

$(YC_PY_LIB): $(YC_OBJS) $(YC_OBJ_DIR)/yask_compiler_api_wrap.o
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) -shared -o $@ $^
	@ls -l $@

#### API tests.

# Build C++ compiler API test.
$(YC_TEST_EXEC): $(YC_TEST_SRC_DIR)/yask_compiler_api_test.cpp $(YC_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) $< $(YC_LFLAGS) -o $@
	@ls -l $@

# Run Python compiler API test to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
py-yc-api-test: $(YC_TEST_SRC_DIR)/yask_compiler_api_test.py $(YC_PY_LIB)
	@echo '*** Running the Python YASK compiler API test...'
	$(RUN_PYTHON) $<
	- dot -Tpdf -O yc-api-test-py.dot
	ls -l yc-api-test-py.*

# Run C++ compiler API test to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
cxx-yc-api-test: $(YC_TEST_EXEC)
	@echo '*** Running the C++ YASK compiler API test...'
	$(RUN_PREFIX) $<
	- dot -Tpdf -O yc-api-test-cxx.dot
	ls -l yc-api-test-cxx.*

# Build C++ compiler API test with exception.
$(YC_TEST_EXEC_WITH_EXCEPTION): $(YC_TEST_SRC_DIR)/yask_compiler_api_exception_test.cpp $(YC_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YC_CXX) $(YC_CXXFLAGS) $< $(YC_LFLAGS) -o $@
	@ls -l $@

# Run Python compiler API test with exceptions to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
py-yc-api-test-with-exception: $(YC_TEST_SRC_DIR)/yask_compiler_api_exception_test.py $(YC_PY_LIB)
	@echo '*** Running the Python YASK compiler API test with exception...'
	$(RUN_PYTHON) $<

# Run C++ compiler API test with exceptions to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
cxx-yc-api-test-with-exception: $(YC_TEST_EXEC_WITH_EXCEPTION)
	@echo '*** Running the C++ YASK compiler API test with exception...'
	$(RUN_PREFIX) $<

######## Misc targets

api-tests:
	$(MAKE) cxx-yc-api-test
	$(MAKE) py-yc-api-test
	$(MAKE) cxx-yc-api-test-with-exception
	$(MAKE) py-yc-api-test-with-exception

all-tests: api-tests

all:
	$(MAKE) compiler
	$(MAKE) api
	$(MAKE) all-tests

old-code-check: $(wildcard $(YC_STENCIL_DIR)/*Stencil.hpp)
	@ for i in $^; do echo "*** WARNING: '$$i' detected. Stencils should now be in .cpp files. If this file is unmodified, just delete it. If modified or new, change its suffix to .cpp."; done

# Remove intermediate files.
clean:
	rm -fv *.s
	rm -fr $(YC_OBJ_DIR)
	rm -fr $(YC_SWIG_OUT_DIR)/build
	rm -fv $(YC_SWIG_OUT_DIR)/*_api_wrap.*
	rm -fv $(YC_OBJS) $(YC_STENCIL_OBJS)
	rm -fv *api-test*.dot* *api-test*.hpp*
	- find . -name '*.o' -print -delete
	- find . -name '*.optrpt' -print -delete

# Remove executables, libs, etc.
realclean: clean
	rm -fr $(YC_OUT_BASE)
	rm -fv $(YC_LIB) $(YC_EXEC)
	rm -fv $(YC_TEST_EXEC) $(YC_TEST_EXEC_WITH_EXCEPTION)
	rm -fv $(YC_PY_MOD)* $(YC_PY_LIB)
	- find . -name '*.pyc' -print -delete
	- find . -name '*~' -print -delete

echo-settings:
	@echo "Build environment for" $(YC_EXEC) on `date`
	@echo host=`hostname`
	@echo YC_CXX=$(YC_CXX)
	@$(YC_CXX) --version
	@echo YC_CXXFLAGS="\"$(YC_CXXFLAGS)\""

help:
	@echo "Example YASK-compiler make commands:"
	@echo " $(MAKE) -j compiler"
	@echo " $(MAKE) -j compiler YC_CXXOPT=-O0"
	@echo " $(MAKE) -j api"
	@echo " $(MAKE) -j all-tests"
	@echo " $(MAKE) clean"
	@echo " $(MAKE) realclean"
