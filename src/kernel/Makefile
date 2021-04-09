##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2021, Intel Corporation
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

# Makefile for the YASK stencil kernel.
# Type 'make help' for usage.

# See top-level YASK Makefile for explanation of vars commonly set via the
# command-line that control kernel configuration and performance.

# Initial defaults.
# These can be overridden on the 'make' command-line.
stencil			?=	iso3dfd
mpi			?=	1
numa			?=	1
pmem			?=	0
real_bytes		?=	4
allow_new_var_types	?=	1
streaming_stores	?=	0
use_ptrs		?=	1
omp			?=	1
offload			?=	0
offload_usm		?=	0
offload_arch		?=	spir64
trace			?=	0
trace_mem		?=	0
check			?=	0
# NB: setting the "trace" and "trace_mem" vars just includes the tracing
# code.  To actually output the trace, you must also use the "-trace"
# run-time option.  Including the "trace" code has a small impact on
# binary size and performance, even if not used. Including the "trace_mem"
# code has a large impact, even if not used.

# Enable offload if offload with unified shared mem is enabled.
# FIXME: offload_usm is not currently enabled.
ifeq ($(offload_usm),1)
 offload		:=	1
endif

# If 'arch' isn't set, determine default architecture by running kernel
# script w/special knob.
# (Do not assume 'yask.sh' has been installed in $(BIN_OUT_DIR) yet.)
arch			?=	$(shell $(BASH) ./yask.sh -show_arch)

# Set 'TARGET' from 'arch', converting codenames and other aliases to ISA names.
# 'TARGET' is the canonical target name.
# The possible values must agree with those in the APIs and YASK compiler.
ifneq ($(filter $(arch),avx snb ivb),)
  TARGET		:=	avx
else ifneq ($(filter $(arch),avx2 hsw bdw),)
  TARGET		:=	avx2
else ifneq ($(filter $(arch),avx512-ymm avx512lo),)
  TARGET		:=	avx512-ymm
else ifneq ($(filter $(arch),avx512 avx512-zmm avx512hi avx512f skx skl clx icx),)
  TARGET		:=	avx512
else ifneq ($(filter $(arch),knl),)
  TARGET		:=	knl
else ifneq ($(filter $(arch),intel64 cpp),)
  TARGET		:=	intel64
else
  $(error Target not recognized; use arch=avx512, avx512-ymm, avx2, avx, knl, or intel64)
endif

# Set the following vars on 'make' cmd-line for corresponding effects:
# - YK_STENCIL and/or YK_ARCH to name binaries differently.
# - YC_STENCIL to override stencil name passed to YASK compiler.
# - YC_TARGET to override target passed to YASK compiler.
YK_STENCIL	?=	$(stencil)
YC_STENCIL	?=	$(stencil)
ifeq ($(offload),1)
  YK_ARCH	:=	$(arch).offload-$(offload_arch)
  YC_TARGET	?=	intel64
else
  YK_ARCH	:=	$(arch)
  YC_TARGET	?=	$(TARGET)
endif

# Default compiler settings based on target.
ifeq ($(TARGET),knl)

 # NUMA_PREF=1 causes YASK to use MCDRAM on a node in "flat" memory mode. It
 # will fall-back to NUMA node 0 in "cache" memory mode. It will *NOT* do
 # the right thing by default in SNC-2 or SNC-4 clustering modes (using
 # either flat or cache memory modes, but these configurations are uncommon.
 ICC_ISA	?=	-xMIC-AVX512
 CXX_ISA	?=	-march=knl
 VEC_MACROS	?=	USE_INTRIN512 USE_RCP28 NUMA_PREF=1

else ifeq ($(TARGET),avx512)

 ICC_ISA	?=	-xCORE-AVX512
 CXX_ISA	?=	-march=skylake-avx512
 VEC_MACROS	?=	USE_INTRIN512

else ifeq ($(TARGET),avx512-ymm)

 ICC_ISA	?=	-xCORE-AVX512
 CXX_ISA	?=	-march=skylake-avx512
 VEC_MACROS	?=	USE_INTRIN512LO

else ifeq ($(TARGET),avx2)

 ICC_ISA	?=	-xCORE-AVX2
 CXX_ISA	?=	-march=haswell
 VEC_MACROS	?=	USE_INTRIN256 USE_AVX2

else ifeq ($(TARGET),avx)

 ICC_ISA	?=	-xAVX
 CXX_ISA	?=	-march=sandybridge
 VEC_MACROS	?=	USE_INTRIN256

else ifeq ($(TARGET),intel64)

 ICC_ISA	?=	-xHOST
 CXX_ISA	?=	-march=native

else

$(error Internal error: arch lists in Makefile are not consistent)

endif # arch-specific.

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# The remainder of this file specifies how to build and test the kernel.
# Vars starting with 'YC_' apply to the stencil compiler.
# Vars starting with 'YK_' apply to the stencil kernel.

# Turn off prefetching at O0 or for offload.
ifneq ($(filter -O0,$(YK_CXXOPT)),)
 pfd_l1		:=	0
 pfd_l2		:=	0
endif
ifeq ($(offload),1)
 pfd_l1		:=	0
 pfd_l2		:=	0
endif

# Flags passed to stencil compiler.
YC_FLAGS	:=	-stencil $(YC_STENCIL) -target $(YC_TARGET) -elem-bytes $(real_bytes)
ifeq ($(use_ptrs),1)
 YC_FLAGS	+=	-use-ptrs
else
 YC_FLAGS	+=	-no-use-ptrs
endif
ifneq ($(fold),)
 YC_FLAGS	+=	-fold $(fold)
endif
ifneq ($(cluster),)
 YC_FLAGS	+=	-cluster $(cluster)
endif
ifneq ($(eqs),)
 YC_FLAGS	+=	-eq-bundles $(eqs)
endif
ifneq ($(radius),)
 YC_FLAGS	+=	-radius $(radius)
endif
ifneq ($(halo),)
 YC_FLAGS	+=	-halo $(halo)
endif
ifneq ($(step_alloc),)
 YC_FLAGS	+=	-step-alloc $(step_alloc)
endif
ifneq ($(domain_dims),)
 YC_FLAGS	+=	-domain-dims $(domain_dims)
endif
ifneq ($(step_dim),)
 YC_FLAGS	+=	-step-dim $(step_dim)
endif
ifneq ($(pfd_l1),)
 YC_FLAGS	+=	-l1-prefetch-dist $(pfd_l1)
endif
ifneq ($(pfd_l2),)
 YC_FLAGS	+=	-l2-prefetch-dist $(pfd_l2)
endif

# Kernel base names.
YK_BASE		:=	yask_kernel
YK_PY_MOD_BASE	:=	$(YK_BASE)

# Kernel input dirs.
YK_SWIG_DIR	:=	$(abspath ./swig)
YK_LIB_SRC_DIR	:=	$(abspath ./lib)
YK_TEST_SRC_DIR	:=	$(abspath ./tests)

# Kernel output dir.
YK_OUT_DIR	:=	$(BUILD_OUT_DIR)/kernel

# Dir for output files dependent on arch but
# not on the stencil.
YK_ARCH_DIR	:=	$(YK_OUT_DIR)/$(YK_ARCH)
YK_OBJ_DIR	:=	$(YK_ARCH_DIR)/obj

# Dirs for stencil-specific output files.
YK_TAG  	:=	$(YK_STENCIL).$(YK_ARCH)
YK_EXT_BASE	:=	$(YK_BASE).$(YK_TAG)
YK_EXT_OUT_DIR	:=	$(YK_OUT_DIR)/$(YK_EXT_BASE)
YK_SWIG_OUT_DIR	:=	$(YK_EXT_OUT_DIR)/swig
YK_EXT_OBJ_DIR	:=	$(YK_EXT_OUT_DIR)/obj
YK_GEN_DIR	:=	$(YK_EXT_OUT_DIR)/gen

# Kernel files.
YK_EXEC		:=	$(BIN_OUT_DIR)/$(YK_EXT_BASE).exe
YK_SCRIPT	:=	$(BIN_OUT_DIR)/yask.sh
YK_LIB		:=	$(LIB_OUT_DIR)/lib$(YK_EXT_BASE)$(SO_SUFFIX)
YK_PY_LIB	:=	$(PY_OUT_DIR)/_$(YK_PY_MOD_BASE)$(SO_SUFFIX)
YK_PY_MOD	:=	$(PY_OUT_DIR)/$(YK_PY_MOD_BASE).py
YK_COMM_SRC_NAMES :=
YK_EXT_SRC_NAMES :=	factory soln_apis context halo stencil_calc setup alloc \
			generic_var yk_var yk_var_apis new_var settings auto_tuner utils
YK_OBJS		:=	$(addprefix $(YK_OBJ_DIR)/,$(addsuffix .o,$(YK_COMM_SRC_NAMES) $(COMM_SRC_NAMES)))
YK_EXT_OBJS	:=	$(addprefix $(YK_EXT_OBJ_DIR)/,$(addsuffix .o,$(YK_EXT_SRC_NAMES)))
YK_CODE_FILE	:=	$(YK_GEN_DIR)/yask_stencil_code.hpp
YK_GEN_HEADERS	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_rank_loops.hpp \
				yask_region_loops.hpp \
				yask_block_loops.hpp \
				yask_mini_block_loops.hpp \
				yask_sub_block_loops.hpp \
				yask_misc_loops.hpp \
				yask_layout_macros.hpp \
				yask_layouts.hpp \
				yask_var_code.hpp) \
				$(YK_CODE_FILE)
YK_INC_DIRS	:=	$(INC_DIRS) $(YK_LIB_SRC_DIR) $(COMM_DIR) $(COEFF_DIR)
YK_INC_GLOB	:=	$(addsuffix /*.hpp,$(YK_INC_DIRS))
YK_API_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_test.exe
YK_API_TEST_EXEC_WITH_EXCEPTION :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_exception_test.exe
YK_VAR_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_var_test.exe
YK_OMP_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_openmp_test.exe
MAKE_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).make-report.txt
YC_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).yask_compiler-report.txt

# File-related macros.
MACROS		+= 	YK_CODE_FILE=$(YK_CODE_FILE)

# YASK loop compiler.
GEN_LOOPS	:=	$(UTILS_BIN_DIR)/gen_loops.pl

# Other scripts and cmds.
GEN_LAYOUTS	:=	$(UTILS_BIN_DIR)/gen_layouts.pl
VIEW_ASM	:=	$(UTILS_BIN_DIR)/view_asm.pl

# Run knobs
YK_SCRIPT_FLAGS	+= "-stencil" $(YK_STENCIL) "-arch" $(YK_ARCH)

# Compiler and default flags.
YK_CXX		:=	icpx
MPI_CXX 	:=	mpiicpc
YK_CXXOPT	?=	-O3
YK_CXXDBG	?=	-g
YK_CXXFLAGS	:=	-std=c++11 -Wall $(YK_CXXDBG) $(YK_CXXOPT)
YK_CXXFLAGS	+=	$(addprefix -I,$(YK_INC_DIRS) $(YK_GEN_DIR))
SWIG_CXXFLAGS	:= 	$(DBL_EPSILON_CXXFLAG)
YK_SO_FLAGS	:=	-shared -fPIC
OMPFLAG 	:=	-fopenmp

# Linker flags.
YK_LIBS		:=	-lrt
YK_LFLAGS	:=	-Wl,-rpath=$(LIB_OUT_DIR) -L$(LIB_OUT_DIR) -l$(YK_EXT_BASE)

# Add options for NUMA.
ifeq ($(numa),1)

# Look for libnuma.
# TODO: make this more portable.
ifneq ($(wildcard /usr/lib64/libnuma.so),)
 YK_LIBS	+=	-lnuma
 MACROS		+=	USE_NUMA
else ifneq ($(wildcard /usr/lib64/libnuma.so.1),)
 YK_LIBS	+=	/usr/lib64/libnuma.so.1
 MACROS		+=	USE_NUMA
endif

# Work-around missing numaif.h:
# IF numaif.h is found in /usr/include,
# THEN enable the macro to use it.
ifneq ($(shell find /usr/include -name 'numaif.h' | wc -l),0)
 MACROS		+=	USE_NUMAIF_H
endif

endif # numa.

# Add options for PMEM.
ifeq ($(pmem),1)
 YK_CXXFLAGS	+=	-I/opt/intel/memkind/include
 YK_LIBS	+=	-L/opt/intel/memkind/lib -lmemkind
 MACROS		+=	USE_PMEM
endif

# Turn on debug macros.
ifeq ($(trace_mem),1)
 MACROS		+=	TRACE TRACE_MEM
else ifeq ($(trace),1)
 MACROS		+=	TRACE
endif
ifeq ($(check),1)
 MACROS		+=	CHECK
endif

# Set MACROS based on individual makefile vars.
ifeq ($(streaming_stores),1)
 MACROS		+=	USE_STREAMING_STORE
endif
MACROS		+=	ALLOW_NEW_VARS=$(allow_new_var_types)

# Default cmd-line args.
ifneq ($(DEF_ARGS),)
 YK_CXXFLAGS	+=	-DDEF_ARGS='"$(DEF_ARGS)"'
endif

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
 VTUNE_GENERIC_DIR	:=	/opt/intel/vtune_profiler
 YK_CXXFLAGS	+=	-I$(VTUNE_GENERIC_DIR)/include
 YK_LIBS	+=	-L$(VTUNE_GENERIC_DIR)/lib64
 ifneq ($(VTUNE_PROFILER_2021_DIR),)
  VTUNE_DIR	:=	$(VTUNE_PROFILER_2021_DIR)
 else ifneq ($(VTUNE_PROFILER_2021_DIR),)
  VTUNE_DIR	:=	$(VTUNE_PROFILER_2021_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_2021_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2021_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_2019_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2019_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_2018_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2018_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_XE_2017_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_XE_2017_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_XE_2016_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_XE_2016_DIR)
 endif
 ifneq ($(VTUNE_DIR),)
  YK_CXXFLAGS	+=	-I$(VTUNE_DIR)/include
  YK_LIBS	+=	-L$(VTUNE_DIR)/lib64
 endif
 YK_LIBS	+=	-littnotify
 YK_VTUNE_CMD	:=	echo "*** Example VTune command:" vtune -c hpc-performance -r vtune.$(YK_TAG) -start-paused $(YK_SCRIPT) $(YK_SCRIPT_FLAGS) "[options]"
endif

# Compiler-specific settings.

# Create a compiler invocation to test for macro settings.
# Use the flags defined so far.
YK_CXX_TEST 	:= 	$(YK_CXX) $(YK_CXXFLAGS) $(EXTRA_YK_CXXFLAGS)
cxx_is_intel_clang :=	$(call MACRO_DEF,$(YK_CXX_TEST),__INTEL_CLANG_COMPILER)
cxx_is_clang	:=	$(call MACRO_DEF,$(YK_CXX_TEST),__clang__)
cxx_is_intel	:=	$(call MACRO_DEF,$(YK_CXX_TEST),__INTEL_COMPILER)
cxx_is_gnu	:= 	$(call MACRO_DEF,$(YK_CXX_TEST),__GNUC__)

# Clang (LLVM-based) Intel compiler (icpx).
ifeq ($(cxx_is_intel_clang),1)
 YK_CXXFLAGS	+=	$(CXX_ISA)
 YK_CXXFLAGS	+=	-ansi-alias \
			-fimf-precision=low -ffp-model=fast -fimf-domain-exclusion=none -fma
 YK_CXXFLAGS	+=	-Wno-unknown-pragmas -Wno-unused-variable \
			-Wno-unused-const-variable -fno-color-diagnostics
 MACROS		+=	NO_VEC2
 VEC_MACROS	+=	NO_SVML
 OMPFLAG	:=	-fiopenmp

# Clang compiler (not Intel version).
else ifeq ($(cxx_is_clang),1)
 YK_CXXFLAGS	+=	$(CXX_ISA)
 YK_CXXFLAGS	+=	-Wno-unknown-pragmas -Wno-unused-variable \
			-Wno-unused-const-variable -fno-color-diagnostics
 VEC_MACROS	+=	NO_SVML

# Classic Intel compiler (not clang-based).
else ifeq ($(cxx_is_intel),1)

 YK_CXXFLAGS	+=	$(ICC_ISA)
 YK_CXXFLAGS	+=	-debug extended -restrict -ansi-alias -fno-alias \
			-fimf-use-svml=true -fimf-precision=low -fast-transcendentals \
			-no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -ftz \
			-fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 YK_CXXFLAGS	+=	-no-diag-message-catalog
 YK_CXXFLAGS	+=	-vec-threshold0
 #YK_CXXFLAGS	+=	-qoverride-limits
 #YK_CXXFLAGS	+=	-Fa
 #YK_CXXFLAGS	+=	-qopt-report=5
 #YK_CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 YK_CXX_VER_CMD :=	$(YK_CXX) -V

# Gnu-compatible compiler (but not Intel or Clang).
else ifeq ($(cxx_is_gnu),1)
 YK_CXXFLAGS	+=	$(CXX_ISA)
 YK_CXXFLAGS	+=	-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable
 SWIG_CXXFLAGS	+=	$(SWIG_GCCFLAGS)
 VEC_MACROS	+=	NO_SVML

# Unknown compiler.
# NB: the ISA will not be specified.
else
 MACROS		+=	NO_VEC NO_UNROLL
 VEC_MACROS	+=	NO_SVML

endif # compiler.

# Add options for OpenMP.
ifeq ($(omp),1)
 OMPFLAGS	:=	$(OMPFLAG)
else
 OMPFLAGS	:=
endif
ifeq ($(offload),1)
 MACROS		+=	USE_OFFLOAD
 ifeq ($(offload_arch),x86_64)
  MACROS	+=	USE_OFFLOAD_X86
 endif
 OMPFLAGS	+=	-fopenmp-targets=$(offload_arch)
 CXX_PREFIX	:=
else
 MACROS		+=	$(VEC_MACROS)
endif
ifeq ($(offload_usm),1)
 MACROS		+=	USE_OFFLOAD_USM
endif

# Add options for MPI.
# MPI settings.
ifeq ($(mpi),1)
 YK_CXXCMD	:=	$(MPI_CXX) -cxx=$(YK_CXX)
 MACROS		+=	USE_MPI
else
 YK_CXXCMD	:=	$(YK_CXX)
 YK_SCRIPT_FLAGS += -ranks 1
endif

# Compile with model_cache=1 or 2 to check prefetching.
# Turn off OpenMP when modeling a cache.
# This is currently not operative! TODO: re-enable cache model.
ifeq ($(model_cache),1)
 MACROS		+=	MODEL_CACHE=1
 OMPFLAGS	:=
else ifeq ($(model_cache),2)
 MACROS		+=	MODEL_CACHE=2
 OMPFLAGS	:=
endif

# Add in final flags and user-added flags.
YK_CXXFLAGS	+=	$(OMPFLAGS) $(addprefix -D,$(MACROS) $(EXTRA_MACROS)) $(EXTRA_YK_CXXFLAGS)
YK_LIBS		+=	$(EXTRA_YK_LIBS)

# Number of stencil/var dims extracted from YASK compiler output.
NSDIMS		:=	`awk '/NUM_STENCIL_DIMS/ {print $$NF}' $(YK_CODE_FILE)`
NVDIMS		:=	`awk '/NUM_STENCIL_AND_VAR_DIMS/ {print $$NF}' $(YK_CODE_FILE)`

######## Loop-compiler configuration:
# The loop indices range from 0..N-1.
# Dim 0 is the step dim, usually time.
# The step loop is handled outside of the generated loops,
# so the following loop codes do not scan over dim 0.
# Dims 1..N-1 are the domain dims, usually spatial.
# Thus, N-1 is the inner-most dim.
# For best perf, this should be the unit-stride dim in the vars.

# Rank loops break up the whole rank into smaller regions.  In order for
# temporal wavefronts to operate properly, the scanning paths must have
# strictly incrementing indices in each dim. Those that do not (e.g.,
# serpentine, square-wave) may *not* be used here when using temporal
# wavefronts. The time loop may be found in
# StencilEquations::run_solution(). The regions are executed sequentially,
# so no OpenMP here.
RANK_LOOP_MODS		?=	nested
RANK_LOOP_FLAGS		?=	-ndims $(NSDIMS) -macroPrefix RANK_
RANK_LOOP_ORDER		?=	1 .. N-1
RANK_LOOP_CODE		?=	$(RANK_LOOP_MODS) loop($(RANK_LOOP_ORDER)) { }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.
REGION_LOOP_OMP		?=	omp parallel for schedule(dynamic,1) proc_bind(spread)
REGION_LOOP_FLAGS	?=	-ndims $(NSDIMS) -macroPrefix REGION_ -ompMod '$(REGION_LOOP_OMP)'
REGION_LOOP_ORDER	?=	1 .. N-1
REGION_LOOP_CODE	?=	$(REGION_LOOP_MODS) omp loop($(REGION_LOOP_ORDER)) { } 

# Block loops break up a block into mini-blocks.
# These are executed sequentially, so no OpenMP.
# Like with rank loops, in order for temporal wavefronts to operate properly,
# the scanning paths must have strictly incrementing indices.
BLOCK_LOOP_MODS		?=	nested
BLOCK_LOOP_FLAGS	?=	-ndims $(NSDIMS) -macroPrefix BLOCK_
BLOCK_LOOP_ORDER	?=	1 .. N-1
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_MODS) loop($(BLOCK_LOOP_ORDER)) { }

# Mini-Block loops break up a mini-block into sub-blocks. The 'omp' modifier
# creates a nested OpenMP loop so that each sub-block is assigned to a
# nested OpenMP thread. The OpenMP construct is not used when running with
# '-bind_block_threads' because another parallel section is created.
MINI_BLOCK_LOOP_OMP	?=	omp parallel for schedule(static,1) proc_bind(spread)
MINI_BLOCK_LOOP_FLAGS	?=	-ndims $(NSDIMS) -macroPrefix MINI_BLOCK_ -ompMod '$(MINI_BLOCK_LOOP_OMP)'
MINI_BLOCK_LOOP_ORDER	?=	1 .. N-1
MINI_BLOCK_LOOP_CODE	?=	$(MINI_BLOCK_LOOP_MODS) omp loop($(MINI_BLOCK_LOOP_ORDER)) { }

# Sub-block loops break up a sub-block into clusters or vectors.  These
# loops are run by a single OMP thread. When offloading, this loop is run in
# parallel on the offload device.  The indexes in this loop
# are 'normalized', i.e., vector units and rank-relative.
SUB_BLOCK_LOOP_FLAGS	?=	-ndims $(NSDIMS) -macroPrefix SUB_BLOCK_
ifeq ($(offload),1)
ifeq ($(offload_arch),x86_64)
 SUB_BLOCK_LOOP_OMP	?=	omp target parallel for device(KernelEnv::_omp_devn)
else
 SUB_BLOCK_LOOP_OMP	?=	omp target teams distribute parallel for device(KernelEnv::_omp_devn)
endif
SUB_BLOCK_LOOP_FLAGS	+=	-ompMod '$(SUB_BLOCK_LOOP_OMP)'
SUB_BLOCK_LOOP_ORDER	?=	2 .. N-1
SUB_BLOCK_LOOP_INNER_DIM ?=	1
SUB_BLOCK_LOOP_CODE	:=	$(SUB_BLOCK_LOOP_MODS) omp loop($(SUB_BLOCK_LOOP_ORDER)) { \
				  loop($(SUB_BLOCK_LOOP_INNER_DIM)) { } }
else
SUB_BLOCK_LOOP_MODS	?=	nested
SUB_BLOCK_LOOP_ORDER	?=	1 .. N-1
SUB_BLOCK_LOOP_CODE	:=	$(SUB_BLOCK_LOOP_MODS) loop($(SUB_BLOCK_LOOP_ORDER)) { }
endif

# General-purpose parallel loop.
MISC_LOOP_OMP		?=	omp parallel for schedule(guided) proc_bind(spread)
MISC_LOOP_FLAGS		?=	-ndims $(NSDIMS) -macroPrefix MISC_ -ompMod '$(MISC_LOOP_OMP)'
MISC_LOOP_ORDER		?=	1 .. N-1
MISC_LOOP_CODE		?=	$(MISC_LOOP_MODS) omp loop($(MISC_LOOP_ORDER)) { }

# Compile rules based on source dir.

$(YK_OBJ_DIR)/%.o: $(COMM_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_OBJ_DIR)/%.o: $(COEFF_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_OBJ_DIR)/%.o: $(YK_LIB_SRC_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_EXT_OBJ_DIR)/%.o: $(YK_LIB_SRC_DIR)/%.cpp $(YK_GEN_HEADERS) $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

######## Primary targets.
# NB: must set stencil and arch make vars to generate the desired YASK kernel.

default: kernel

# Compile the kernel.
kernel: $(YK_EXEC) $(YK_SCRIPT) $(MAKE_REPORT_FILE)
	@echo "*** Binary" $(YK_EXEC) "has been built."; \
	echo "*** See" $(MAKE_REPORT_FILE) "for build information."; \
	echo "*** Run command:" $(YK_SCRIPT) $(YK_SCRIPT_FLAGS) "[options]"
	@$(YK_VTUNE_CMD)

$(YK_LIB): $(YK_OBJS) $(YK_EXT_OBJS)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $(YK_SO_FLAGS) -o $@ $^ $(YK_LIBS)
	@ls -l $@

$(YK_EXEC): yask_main.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

$(YK_SCRIPT): ./yask.sh
	$(MKDIR) $(dir $@)
	cp $< $@
	chmod a+rx $@
	@ls -l $@

scripts: $(YK_SCRIPT)

$(MAKE_REPORT_FILE): $(YK_LIB)
	$(MKDIR) $(dir $@)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	- $(MAKE) echo-settings >> $@ 2>&1

# Generated source files.
# NB: YK_CODE_FILE is a dependency on loop and layout code because
# the number of dims from that file affects the loops and layouts.
$(YK_GEN_DIR)/yask_rank_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(RANK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_RANK_LOOP_FLAGS) "$(RANK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_region_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(REGION_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_REGION_LOOP_FLAGS) "$(REGION_LOOP_CODE)"

$(YK_GEN_DIR)/yask_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_BLOCK_LOOP_FLAGS) "$(BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_mini_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(MINI_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_MINI_BLOCK_LOOP_FLAGS) "$(MINI_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_sub_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(SUB_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_SUB_BLOCK_LOOP_FLAGS) "$(SUB_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_misc_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(MISC_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_MISC_LOOP_FLAGS) "$(MISC_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -m $(NVDIMS) > $@
	@- $(INDENT) $@

$(YK_GEN_DIR)/yask_layouts.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -d $(NVDIMS) > $@
	@- $(INDENT) $@

$(YK_GEN_DIR)/yask_var_code.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -v $(NVDIMS) > $@

# NB: '$(BASH) -o pipefail' ensures failure of command before '| tee' is visible.
$(YK_CODE_FILE): $(YC_EXEC)
	$(MKDIR) $(dir $@)
	@- rm -f $@
	$(BASH) -o pipefail -c \
	  "$(RUN_PREFIX) $< $(YC_FLAGS) -p $@ $(EXTRA_YC_FLAGS) 2>&1 | tee $(YC_REPORT_FILE)"
	@- $(INDENT) $@

headers: $(YK_GEN_HEADERS)
	@ echo 'Header files generated.'

######## API targets
# NB: must set stencil and arch to generate the desired kernel API.

# Build C++ and Python kernel API libs.
api: $(YK_LIB) $(YK_PY_LIB)

# Build python kernel API lib.
py-api: $(YK_PY_LIB)

# Build C++ SWIG wrapper code.
# This also creates $(YK_PY_MOD) as a side-effect.
$(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.cpp: $(YK_SWIG_DIR)/yask*.i $(INC_DIR)/*.hpp
	$(SWIG) -version
	$(MKDIR) $(YK_SWIG_OUT_DIR) $(PY_OUT_DIR)
	$(SWIG) -v -DYK_MODULE=$(YK_PY_MOD_BASE) -cppext cpp \
	  -DUSE_MPI -DMPI_VERSION=3 -DMPI_Comm=int \
	  -I$(INC_DIR) -I$(COMM_DIR) -I$(COMM_DIR)/swig -I$(COEFF_DIR) \
	  -c++ -python -o $@ -outdir $(PY_OUT_DIR) -builtin $<

$(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o: $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.cpp
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ $(SWIG_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<
	@ls -l $@

$(YK_PY_LIB): $(YK_OBJS)  $(YK_EXT_OBJS) $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $(YK_SO_FLAGS) -o $@ $^ $(YK_LIBS)
	@ls -l $@

######## Misc targets

# Generate the code file using the built-in compiler.
code-file: $(YK_CODE_FILE)

# Just the common obj files.
common-objs: $(YK_OBJS)

# Special targets to avoid running stencil compiler and replacing the stencil-code file.
# NB: This trick is only needed when using the compiler API to create
# a stencil to replace the one normally created by the pre-built stencil compiler
# AND when the compiler binary doesn't exist or can't be found.
NO_YC_MAKE_FLAGS := --new-file=$(YK_CODE_FILE)
kernel-no-yc:
	$(MAKE) $(NO_YC_MAKE_FLAGS) kernel

api-no-yc:
	$(MAKE) $(NO_YC_MAKE_FLAGS) api

py-api-no-yc:
	$(MAKE) $(NO_YC_MAKE_FLAGS) py-api

# Remove intermediate files.
# Should not trigger remake of stencil compiler.
# Make this target before rebuilding YASK with any new parameters.
clean:
	rm -fr $(YK_OUT_DIR)
	rm -fv *.s
	- find . $(COMM_DIR) $(COEFF_DIR) -name '*.*.o' -print -delete
	- find . $(COMM_DIR) $(COEFF_DIR) -name '*.optrpt' -print -delete

# Remove executables, libs, etc.
# Also remove logs from kernel dir, which are most likely from testing.
realclean: clean
	rm -fv $(YK_LIB) $(YK_EXEC) $(MAKE_REPORT_FILE)
	rm -fv $(YK_PY_MOD)* $(YK_PY_LIB)
	rm -fv $(YK_API_TEST_EXEC) $(YK_API_TEST_EXEC_WITH_EXCEPTION)
	rm -fv $(BUILD_OUT_DIR)/*report.txt
	- find . -name '*.pyc' -print -delete
	- find . -name '*~' -print -delete

echo-settings:
	@echo "Build environment for" $(YK_EXEC) on `date`; \
	echo TARGET=$(TARGET); \
	echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""; \
	echo BLOCK_LOOP_FLAGS="\"$(BLOCK_LOOP_FLAGS)\""; \
	echo BLOCK_LOOP_ORDER="\"$(BLOCK_LOOP_ORDER)\""; \
	echo BLOCK_LOOP_MODS="\"$(BLOCK_LOOP_MODS)\""; \
	echo DEF_ARGS="\"$(DEF_ARGS)\""; \
	echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""; \
	echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""; \
	echo EXTRA_YK_CXXFLAGS="\"$(EXTRA_YK_CXXFLAGS)\""; \
	echo ISA=$(ISA); \
	echo MACROS="\"$(MACROS)\""; \
	echo MINI_BLOCK_LOOP_CODE="\"$(MINI_BLOCK_LOOP_CODE)\""; \
	echo MINI_BLOCK_LOOP_FLAGS="\"$(MINI_BLOCK_LOOP_FLAGS)\""; \
	echo MINI_BLOCK_LOOP_ORDER="\"$(MINI_BLOCK_LOOP_ORDER)\""; \
	echo MINI_BLOCK_LOOP_MODS="\"$(MINI_BLOCK_LOOP_MODS)\""; \
	echo MISC_LOOP_CODE="\"$(MISC_LOOP_CODE)\""; \
	echo MISC_LOOP_FLAGS="\"$(MISC_LOOP_FLAGS)\""; \
	echo MISC_LOOP_ORDER="\"$(MISC_LOOP_ORDER)\""; \
	echo MISC_LOOP_MODS="\"$(MISC_LOOP_MODS)\""; \
	echo OMPFLAGS="\"$(OMPFLAGS)\""; \
	echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""; \
	echo RANK_LOOP_FLAGS="\"$(RANK_LOOP_FLAGS)\""; \
	echo RANK_LOOP_ORDER="\"$(RANK_LOOP_ORDER)\""; \
	echo RANK_LOOP_MODS="\"$(RANK_LOOP_MODS)\""; \
	echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""; \
	echo REGION_LOOP_FLAGS="\"$(REGION_LOOP_FLAGS)\""; \
	echo REGION_LOOP_ORDER="\"$(REGION_LOOP_ORDER)\""; \
	echo REGION_LOOP_MODS="\"$(REGION_LOOP_MODS)\""; \
	echo SUB_BLOCK_LOOP_CODE="\"$(SUB_BLOCK_LOOP_CODE)\""; \
	echo SUB_BLOCK_LOOP_FLAGS="\"$(SUB_BLOCK_LOOP_FLAGS)\""; \
	echo SUB_BLOCK_LOOP_ORDER="\"$(SUB_BLOCK_LOOP_ORDER)\""; \
	echo SUB_BLOCK_LOOP_MODS="\"$(SUB_BLOCK_LOOP_MODS)\""; \
	echo YASK_OUTPUT_DIR=$(YASK_OUTPUT_DIR); \
	echo YC_FLAGS="\"$(YC_FLAGS)\""; \
	echo YC_STENCIL=$(YC_STENCIL); \
	echo YC_TARGET=$(YC_TARGET); \
	echo YK_ARCH=$(YK_ARCH); \
	echo YK_BASE=$(YK_BASE); \
	echo YK_CXX=$(YK_CXX); \
	echo YK_CXXCMD=$(YK_CXXCMD); \
	echo YK_CXXDBG=$(YK_CXXDBG); \
	echo YK_CXXFLAGS="\"$(YK_CXXFLAGS)\""; \
	echo YK_CXXOPT=$(YK_CXXOPT); \
	echo YK_EXT_TAG=$(YK_EXT_TAG); \
	echo YK_STENCIL=$(YK_STENCIL); \
	echo YK_TAG=$(YK_TAG); \
	echo arch=$(arch); \
	echo cluster=$(cluster); \
	echo fold=$(fold); \
	echo offload=$(offload); \
	echo offload_usm=$(offload_usm); \
	echo omp_block_schedule=$(omp_block_schedule); \
	echo omp_misc_schedule=$(omp_misc_schedule); \
	echo omp_region_schedule=$(omp_region_schedule); \
	echo pfd_l1=$(pfd_l1); \
	echo pfd_l2=$(pfd_l2); \
	echo radius=$(radius); \
	echo real_bytes=$(real_bytes); \
	echo stencil=$(stencil); \
	echo streaming_stores=$(streaming_stores)
	git status
	uname -a
	$(YK_CXX) --version

# Print stats on inner SIMD loops from asm file.
code-stats: $(YK_LIB)
	$(MKDIR) $(YK_OBJ_DIR)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -S -o $(YK_OBJ_DIR)/factory.s $(YK_LIB_SRC_DIR)/factory.cpp
	@echo "Code stats for stencil computation:"
	$(PERL) $(VIEW_ASM) -p -l -f='calc_.*opt' $(YK_OBJ_DIR)/factory.s

# Print some usage info.
help:
	@echo "Example builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd           # Build iso3dfd stencil with default settings"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg               # Build ssg stencil with default settings"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd radius=4  # Specify stencil radius"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd radius=2 YK_STENCIL=iso3dfd_r2 # Specify binary stencil name"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd arch=knl  # Specify arch target"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg fold='x=1,y=2,z=4' # Specify folding (data layout)"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg cluster='x=2' # Specify clustering (unrolling)"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg real_bytes=8  # Use 8-byte (double) floats"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg mpi=0         # No MPI support"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp vtune=1       # Add VTune start/stop instrumentation"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=fsg YK_CXXOPT='-O2' # Use O2 optimization"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=fsg YK_CXX=icpc   # Use classic Intel compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg YK_CXX=g++    # Use gnu compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg MPI_CXX=mpiCC # Specify MPI compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp YK_CXX=icpc EXTRA_YK_CXXFLAGS='-qopt-report -Fa' # Emit opt-report and asm file"; \
	echo " "
	@echo "Example builds of kernel API for C++ and Python apps:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd yk-api"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp yk-api"; \
	echo " "
	@echo "Example debug builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd mpi=0 omp=0 YK_CXXOPT='-O0' check=1 # No optimization, OpenMP, or MPI; use internal checking"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=avx2 stencil=test_2d YK_CXXOPT='-O0' trace=1   # Enable tracing; run with '-trace' to get trace"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=ssg radius=0 fold='x=1,y=1,z=1' YK_CXXOPT='-O0' trace_mem=1 # Trace all mem accesses with '-trace'"; \
	echo " "
	@echo "Example regression tests (run before any git push or pull request):"; \
	echo " $(MAKE) -j all                   # Normal full API and stencil tests"; \
	echo " $(MAKE) -j all ranks=8           # Test with 8 MPI ranks"; \
	echo " $(MAKE) -j all YK_CXXOPT=-O1     # Test at O1 optimization"; \
	echo " $(MAKE) -j all YK_CXX=g++ mpi=0  # Test g++ w/o MPI"; \
	echo " $(MAKE) -j all YASK_OUTPUT_DIR=./test  # Test using alternate output dir"


#################################
########### Tests ###############
#################################
# TODO: convert all testing to a separate test framework.

# Disable folding and clustering for offload testing.
empty		:=
space		:=	$(empty) $(empty)
comma		:=	,
ifeq ($(offload),1)
FOLD		=
CLUSTER		=
else
FOLD		=	fold=$(subst $(space),$(comma),$(1))
CLUSTER		=	cluster=$(subst $(space),$(comma),$(1))
endif

TEST_ARGS	:=	real_bytes=8 allow_new_var_types=0 check=1 trace=1
TEST_MAKE 	:=	$(MAKE) $(TEST_ARGS)

# Default number of ranks for running tests.
# 4 is good because it tests in-plane diagonal exchanges for 2D and 3D tests.
# 8 would test all exchanges for 3D tests.
ifneq ($(mpi),1)
ranks	:=	1
else
ranks	:=	4
endif

# Default min & max test number to run.
first_test	:=	0
last_test	:=	999

# Default threads.
max_threads	:=	6
block_threads	:=	2

### Unit tests.

$(YK_OMP_TEST_EXEC): $(YK_TEST_SRC_DIR)/openmp_test.cpp
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< -o $@ $(YK_LIBS)
	@ls -l $@

cxx-yk-omp-test:
	rm -f $(YK_OMP_TEST_EXEC)
	$(TEST_MAKE) $(YK_OMP_TEST_EXEC)
	@echo '*** Running the C++ YASK OpenMP test...'
	$(RUN_PREFIX) $(YK_OMP_TEST_EXEC) $(max_threads)

$(YK_VAR_TEST_EXEC): $(YK_TEST_SRC_DIR)/var_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

cxx-yk-var-test:
	rm -f $(YK_OMP_VAR_EXEC)
	$(TEST_MAKE) $(YK_VAR_TEST_EXEC)
	@echo '*** Running the C++ YASK var test...'
	$(RUN_PREFIX) $(YK_VAR_TEST_EXEC)

### API tests.

# Build C++ kernel tests.
$(YK_API_TEST_EXEC): $(YK_TEST_SRC_DIR)/yask_kernel_api_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

# Build and run C++ tests.
cxx-yk-api-test:
	$(TEST_MAKE) $(YK_API_TEST_EXEC) allow_new_var_types=1
	@echo '*** Running the C++ YASK kernel API test...'
	$(RUN_PREFIX) $(YK_API_TEST_EXEC)

# Run Python kernel API test.
py-yk-api-test: $(YK_TEST_SRC_DIR)/yask_kernel_api_test.py $(YK_PY_LIB)
	@echo '*** Running the Python YASK kernel API test...'
	$(RUN_PYTHON) $<

# Build C++ kernel tests with exceptions.
$(YK_API_TEST_EXEC_WITH_EXCEPTION): $(YK_TEST_SRC_DIR)/yask_kernel_api_exception_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

# Run C++ tests with exceptions.
cxx-yk-api-test-with-exception: $(YK_API_TEST_EXEC_WITH_EXCEPTION)
	@echo '*** Running the C++ YASK kernel API test with exception...'
	$(RUN_PREFIX) $<

# Run Python kernel API test with exceptions.
py-yk-api-test-with-exception: $(YK_TEST_SRC_DIR)/yask_kernel_api_exception_test.py $(YK_PY_LIB)
	@echo '*** Running the Python YASK kernel API test with exception...'
	$(RUN_PYTHON) $<

### API tests for the compiler.
# These targets run the tests in the compiler directory,
# then they move the output files into the correct location
# for the kernel.

# Run Python compiler API test to create stencil-code file.
py-yc-api-test:
	$(MAKE) -C $(YC_SRC_DIR) $@
	$(MKDIR) $(YK_GEN_DIR)
	mv $(YC_SRC_DIR)/yc-api-test-py.hpp $(YK_CODE_FILE)

# Run C++ compiler API test to create stencil-code file.
cxx-yc-api-test:
	$(MAKE) -C $(YC_SRC_DIR) $@
	$(MKDIR) $(YK_GEN_DIR)
	mv $(YC_SRC_DIR)/yc-api-test-cxx.hpp $(YK_CODE_FILE)

# Validation runs for each binary.
DEF_TEST_ARGS := -log_dir $(TEST_LOG_OUT_DIR) \
	-ranks $(ranks) -stencil $(YK_STENCIL) -arch $(YK_ARCH) -v \
	-trial_steps 2 -max_threads $(max_threads) -block_threads $(block_threads) \
	-ep 0 -mp 0 -no-allow_addl_padding

# Test args for 1 and >1 ranks.
test_args0	:=	$(DEF_TEST_ARGS) -l 48 -r 32 -b 24 -rt 0 $(EXTRA_TEST_ARGS)
test_args1	:=	$(DEF_TEST_ARGS) -l 48 -r 32 -b 24 -mb 16 -rt 1 $(EXTRA_TEST_ARGS)
test_args2	:=	$(DEF_TEST_ARGS) -l 63 -r 32 -b 24 -mb 16 -rt 2 $(EXTRA_TEST_ARGS)
test_args3	:=	$(DEF_TEST_ARGS) -l 63 -r 48 -b 24 -mb 16 -bt 2 $(EXTRA_TEST_ARGS)
test_args4	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -bind_block_threads $(EXTRA_TEST_ARGS)

# Test args for >1 ranks only.
test_args10	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -no-use_shm -overlap_comms $(EXTRA_TEST_ARGS)
ifeq ($(offload),1)
test_args11	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -no-use_shm -no-overlap_comms $(EXTRA_TEST_ARGS)
else
test_args11	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -use_shm -no-overlap_comms $(EXTRA_TEST_ARGS)
endif

# Run the kernel binary using several combos of sizes and ranks.
yk-tests:
	if (( $(first_test) <= 0 && $(last_test) >= 0 )); then $(YK_SCRIPT) $(test_args0); fi
	if (( $(first_test) <= 1 && $(last_test) >= 1 )); then $(YK_SCRIPT) $(test_args1); fi
	if (( $(first_test) <= 2 && $(last_test) >= 2 )); then $(YK_SCRIPT) $(test_args2); fi
	if (( $(first_test) <= 3 && $(last_test) >= 3 )); then $(YK_SCRIPT) $(test_args3); fi
	if (( $(first_test) <= 4 && $(last_test) >= 4 )); then $(YK_SCRIPT) $(test_args4); fi

# These are only run w/>1 rank.
yk-mpi-tests:
	if (( $(first_test) <= 10 && $(last_test) >= 10 )); then $(YK_SCRIPT) $(test_args10); fi
	if (( $(first_test) <= 11 && $(last_test) >= 11 )); then $(YK_SCRIPT) $(test_args11); fi

# Run the default YASK compiler and kernel.
# First run on 1 rank, then multiple ranks if ranks>1.
yc-and-yk-test: $(YK_EXEC) $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests yk-mpi-tests; fi

# Run the YASK kernel test without implicity using the YASK compiler.
yk-test-no-yc: kernel-no-yc $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests yk-mpi-tests; fi

# Run the kernel API tests for C++ and Python with and w/o expected exceptions.
api-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-api-test real_bytes=8 stencil=iso3dfd
	$(MAKE) clean; $(MAKE) py-yk-api-test stencil=iso3dfd
	$(MAKE) clean; $(MAKE) cxx-yk-api-test-with-exception real_bytes=8 stencil=iso3dfd
	$(MAKE) clean; $(MAKE) py-yk-api-test-with-exception stencil=iso3dfd

# Run several stencils using built-in validation.
# NB: set arch var as applicable.
# NB: save some compile time by using YK_CXXOPT=-O2 or -O1.
# These tests are focused on the kernel and not the compiler.
# For testing both the kernel and compiler in various combinations,
# run the tests from the top-level Makefile.
STENCIL_TEST	:=	$(TEST_MAKE) yc-and-yk-test

1d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d $(call FOLD,x=4) $(call CLUSTER,x=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_func_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_step_cond_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_boundary_1d $(call FOLD,x=4)

2d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_empty step_dim=t domain_dims=d1,d2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_empty_2d
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d $(call FOLD,x=2 y=2) $(call CLUSTER,y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d $(call FOLD,x=2 y=2) mpi=0 ranks=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d arch=intel64 $(call FOLD,x=3 y=2) domain_dims=y,x
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_reverse_2d radius=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_2d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d EXTRA_YC_FLAGS=-interleave-misc
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_2d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_2d $(call FOLD,x=2 y=2)

3d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d $(call FOLD,x=2 y=2 z=2) EXTRA_YC_FLAGS=-fus
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d $(call FOLD,x=2 y=2 z=2) EXTRA_YC_FLAGS=-lus
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d $(call FOLD,x=2 z=2) domain_dims=z,y,x
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_3d $(call FOLD,y=2 x=2) domain_dims=x,z,y
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_partial_3d $(call FOLD,x=2 z=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_partial_3d $(call FOLD,x=2 z=2) domain_dims=x,z,y
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stream_3d radius=5
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_3d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_3d $(call FOLD,x=2 z=2)

# 3D tests w/specific shapes.
3d-tests2:
	$(MAKE) clean; $(STENCIL_TEST) stencil=3axis $(call FOLD,x=2 y=2) $(call CLUSTER,x=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=3axis $(call FOLD,x=2 y=2) $(call CLUSTER,z=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=3axis_with_diags $(call FOLD,x=2 z=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=3plane $(call FOLD,y=2 z=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=cube $(call FOLD,x=2 y=2)

# 3D tests w/actual seismic stencils.
3d-tests3:
	$(MAKE) clean; $(STENCIL_TEST) stencil=iso3dfd radius=3 $(call FOLD,x=2 y=2) domain_dims=z,x,y
	$(MAKE) clean; $(STENCIL_TEST) stencil=iso3dfd_sponge radius=6 $(call FOLD,x=2 z=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=ssg $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=ssg_merged $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=awp $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=awp_abc $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=tti $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg_abc $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg_merged $(call FOLD,x=2 y=2)

4d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_4d $(call FOLD,w=2 x=2)

stencil-tests:
	$(MAKE) 1d-tests
	$(MAKE) 2d-tests
	$(MAKE) 3d-tests
	$(MAKE) 3d-tests2
	$(MAKE) 3d-tests3
	$(MAKE) 4d-tests

unit-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-omp-test
	$(MAKE) clean; $(MAKE) cxx-yk-var-test stencil=test_3d $(call FOLD,x=2 y=2)

all-tests:
	$(MAKE) unit-tests
	$(MAKE) stencil-tests
	$(MAKE) api-tests

# First, build a kernel and an API lib w/o any validation just to test the build env.
# Then, build and run all the tests.
all:
	$(MAKE) kernel stencil=test_3d
	$(MAKE) api stencil=test_3d
	$(MAKE) all-tests

