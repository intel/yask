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

# Makefile for the YASK stencil kernel.
# Type 'make help' for usage.

# See top-level YASK Makefile for explanation of vars commonly set via the
# command-line that control kernel configuration and performance.

# Initial defaults.
# These can be overridden on the 'make' command-line.
stencil			?=	iso3dfd
omp			?=	1
mpi			?=	1
offload			?=	0
offload_arch		?=	spir64
numa			?=	1

real_bytes		?=	4
allow_new_var_types	?=	1
streaming_stores	?=	0
use_ptrs		?=	1
use_safe_ptrs		?=	0
outer_domain_layout	?=	0
inner_misc_layout 	?=	1
early_loads		?=	0
min_buffer_len		?=	1
read_ahead_dist		?=	0

trace			?=	0
trace_mem		?=	0
check			?=	0
# NB: setting the "trace" and "trace_mem" vars just includes the tracing
# code.  To actually output the trace, you must also use the "-trace"
# run-time option.  Including the "trace" code has a small impact on
# binary size and performance, even if not used. Including the "trace_mem"
# code has a large impact, even if not used.

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
else ifneq ($(filter $(arch),avx512 avx512-zmm avx512hi avx512f skx skl clx icx spr),)
  TARGET		:=	avx512
else ifneq ($(filter $(arch),knl),)
  TARGET		:=	knl
else ifneq ($(filter $(arch),intel64 cpp),)
  TARGET		:=	intel64
else
  $(error Target not recognized; use arch=avx512, avx512-ymm, avx2, avx, knl, or intel64)
endif

# Enable offload if offload with unified shared mem is enabled.
offload_usm		?=	0
ifeq ($(offload_usm),1)
 offload		:=	1
endif

# YASK compiler settings for offload.
ifeq ($(offload),1)
 pfd_l1		:=	0
 pfd_l2		:=	0
 inner_loop_dim :=	1
 early_loads	:=	1
 outer_domain_layout	:=	1
endif

# Main vars for naming the libraries and executables and running
# the YASK compiler.
# Set the following vars on 'make' cmd-line for corresponding effects:
# - YK_STENCIL and/or YK_ARCH to name libraries and executables differently.
# - YC_STENCIL to override stencil name passed to YASK compiler.
# - YC_TARGET to override target passed to YASK compiler.
YK_STENCIL	?=	$(stencil)$(YK_STENCIL_SUFFIX)
YC_STENCIL	?=	$(stencil)
ifeq ($(offload),1)
  YK_ARCH	:=	$(arch).offload-$(offload_arch)
  YC_TARGET	?=	intel64
else
  YK_ARCH	:=	$(arch)
  YC_TARGET	?=	$(TARGET)
endif
YK_TAG  	:=	$(YK_STENCIL).$(YK_ARCH)

# Default settings based on target.
ifeq ($(TARGET),knl)

 # NUMA_PREF=1 causes YASK to use MCDRAM on a node in "flat" memory mode. It
 # will fall-back to NUMA node 0 in "cache" memory mode. It will *NOT* do
 # the right thing by default in SNC-2 or SNC-4 clustering modes (using
 # either flat or cache memory modes, but these configurations are uncommon.
 ICC_ISA	?=	-xMIC-AVX512
 GXX_ISA	?=	-march=knl
 VEC_MACROS	?=	USE_INTRIN512 USE_RCP28
 MACROS		+=	NUMA_PREF=1
 pfd_l1		?=	1

else ifeq ($(TARGET),avx512)

 ICC_ISA	?=	-xCORE-AVX512
 GXX_ISA	?=	-mavx512f -mavx512vl -mavx512cd -mavx512bw
 VEC_MACROS	?=	USE_INTRIN512

else ifeq ($(TARGET),avx512-ymm)

 ICC_ISA	?=	-xCORE-AVX512
 GXX_ISA	?=	-mavx512f -mavx512vl -mavx512cd -mavx512bw
 VEC_MACROS	?=	USE_INTRIN512LO

else ifeq ($(TARGET),avx2)

 ICC_ISA	?=	-xCORE-AVX2
 GXX_ISA	?=	-mavx2
 VEC_MACROS	?=	USE_INTRIN256 USE_AVX2

else ifeq ($(TARGET),avx)

 ICC_ISA	?=	-xAVX
 GXX_ISA	?=	-mavx
 VEC_MACROS	?=	USE_INTRIN256

else ifeq ($(TARGET),intel64)

 ICC_ISA	?=	-xHOST
 GXX_ISA	?=

else

$(error Internal error: arch lists in Makefile are not consistent)

endif # arch-specific.

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# The remainder of this file specifies how to build and test the kernel.
# Vars starting with 'YC_' apply to the stencil compiler.
# Vars starting with 'YK_' apply to the stencil kernel.

# Flags passed to stencil compiler.
YC_FLAGS	:=	-stencil $(YC_STENCIL) -target $(YC_TARGET) -elem-bytes $(real_bytes)
YC_FLAGS	+=	-print-eqs

# Stencil compiler flags that take an argument.
ifneq ($(radius),)
 YC_FLAGS	+=	-radius $(radius)
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
ifneq ($(inner_loop_dim),)
 YC_FLAGS	+=	-inner-loop-dim $(inner_loop_dim)
endif
ifneq ($(min_buffer_len),)
 YC_FLAGS	+=	-min-buffer-len $(min_buffer_len)
endif
ifneq ($(read_ahead_dist),)
 YC_FLAGS	+=	-read-ahead-dist $(read_ahead_dist)
endif

# Stencil compiler flags that are boolean.
ifeq ($(use_ptrs),1)
 YC_FLAGS	+=	-use-ptrs
else
 YC_FLAGS	+=	-no-use-ptrs
endif
ifeq ($(use_safe_ptrs),1)
 YC_FLAGS	+=	-use-safe-ptrs
else
 YC_FLAGS	+=	-no-use-safe-ptrs
endif
ifeq ($(outer_domain_layout),1)
 YC_FLAGS	+=	-outer-domain-layout
else
 YC_FLAGS	+=	-no-outer-domain-layout
endif
ifeq ($(inner_misc_layout),1)
 YC_FLAGS	+=	-inner-misc-layout
else
 YC_FLAGS	+=	-no-inner-misc-layout
endif
ifeq ($(early_loads),1)
 YC_FLAGS	+=	-early-loads
else
 YC_FLAGS	+=	-no-early-loads
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
YK_SOLUTION_FILE	:=	$(YK_GEN_DIR)/yask_solution_code.hpp
YK_LOOP_FILES	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_rank_loops.hpp \
				yask_mega_block_loops.hpp \
				yask_block_loops.hpp \
				yask_micro_block_loops.hpp \
				yask_nano_block_loops.hpp \
				yask_pico_block_loops.hpp \
				yask_misc_loops.hpp)
YK_GEN_HEADERS	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_layout_macros.hpp \
				yask_layouts.hpp \
				yask_var_code.hpp) \
				$(YK_LOOP_FILES) \
				$(YK_CODE_FILE) \
				$(YK_SOLUTION_FILE)
YK_INC_DIRS	:=	$(INC_DIRS) $(YK_LIB_SRC_DIR) $(COMM_DIR) $(COEFF_DIR)
YK_INC_GLOB	:=	$(addsuffix /*.hpp,$(YK_INC_DIRS))
YK_API_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_test.exe
YK_API_TEST_EXEC_WITH_EXCEPTION :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_exception_test.exe
YK_VAR_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_var_test.exe
YK_OMP_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_openmp_test.exe
MAKE_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).make-report.txt
YC_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).yask_compiler-report.txt

# File-related macros.
MACROS		+= 	YK_CODE_FILE=$(YK_CODE_FILE) YK_SOLUTION_FILE=$(YK_SOLUTION_FILE)

# YASK loop compiler.
GEN_LOOPS	:=	$(UTILS_BIN_DIR)/gen_loops.pl

# Other scripts and cmds.
GEN_LAYOUTS	:=	$(UTILS_BIN_DIR)/gen_layouts.pl
VIEW_ASM	:=	$(UTILS_BIN_DIR)/view_asm.pl

# Run knobs
YK_SCRIPT_FLAGS	+= "-stencil" $(YK_STENCIL) "-arch" $(YK_ARCH)

# Compiler and default flags.
YK_CXX		:=	$(CXX)
MPI_CXX 	:=	mpiicpc
YK_CXXOPT	?=	-O3
YK_CXXDBG	?=	-g
YK_CXXFLAGS	:=	-std=c++17
YK_CXXWARN	:=	-Wall
YK_CXX_INCFLAGS	:=	$(addprefix -I,$(YK_INC_DIRS) $(YK_GEN_DIR))
YK_CXXFLAGS	+=	$(YK_CXX_INCFLAGS)
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

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
 VTUNE_GENERIC_DIR	:=	/opt/intel/vtune_profiler
 YK_CXXFLAGS	+=	-I$(VTUNE_GENERIC_DIR)/include
 YK_LIBS	+=	-L$(VTUNE_GENERIC_DIR)/lib64
 ifneq ($(VTUNE_PROFILER_DIR),)
  VTUNE_DIR	:=	$(VTUNE_PROFILER_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_2021_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2021_DIR)
 else ifneq ($(VTUNE_AMPLIFIER_2020_DIR),)
  VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2020_DIR)
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
cxx_is_llvm_intel :=	$(call MACRO_DEF,$(YK_CXX_TEST),__INTEL_LLVM_COMPILER)
cxx_is_clang	:=	$(call MACRO_DEF,$(YK_CXX_TEST),__clang__)
cxx_is_intel	:=	$(call MACRO_DEF,$(YK_CXX_TEST),__INTEL_COMPILER)
cxx_is_gnu	:= 	$(call MACRO_DEF,$(YK_CXX_TEST),__GNUC__)

# LLVM-based Intel compiler (icpx).
ifeq ($(cxx_is_llvm_intel),1)
 YK_CXXFLAGS	+=	$(GXX_ISA) $(ICC_ISA)
 YK_CXXDBG	+=	-gline-tables-only -fdebug-info-for-profiling
 YK_CXXFLAGS	+=	-ansi-alias \
			-fimf-precision=low -fp-model fast -fimf-domain-exclusion=none -fma
 YK_CXXWARN	+=	-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable \
			-Wno-unused-const-variable -fno-color-diagnostics
 OMPFLAG	:=	-fiopenmp
 SWIG_CXXFLAGS	+=	-Wno-deprecated-declarations
 MACROS		+=	INTEL_OMP
 VEC_MACROS	+=	NO_PRAGMA_VEC2

# Clang compiler (clang++, not Intel version).
else ifeq ($(cxx_is_clang),1)
 YK_CXXFLAGS	+=	$(GXX_ISA)
 YK_CXXWARN	+=	-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable \
			-Wno-unused-const-variable -fno-color-diagnostics
 VEC_MACROS	+=	NO_SVML NO_PRAGMA_VEC2

# Classic Intel compiler (icpc, not LLVM-based).
else ifeq ($(cxx_is_intel),1)

 YK_CXXFLAGS	+=	$(ICC_ISA)
 YK_CXXDBG	+=	-debug extended
 YK_CXXFLAGS	+=	-restrict -ansi-alias -fno-alias \
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
 OMPFLAG	:=	-qopenmp

# Gnu-compatible compiler (g++, not Intel or LLVM).
else ifeq ($(cxx_is_gnu),1)
 YK_CXXFLAGS	+=	$(GXX_ISA)
 YK_CXXWARN	+=	-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable
 SWIG_CXXFLAGS	+=	$(SWIG_GCCFLAGS)
 VEC_MACROS	+=	NO_SVML

# Unknown compiler.
# NB: the ISA will not be specified.
else
 MACROS		+=	NO_PRAGMA_VEC1 NO_PRAGMA_VEC2 NO_PRAGMA_UNROLL
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
YK_MACROS	:=	$(addprefix -D,$(MACROS) $(subst $(comma),$(space),$(EXTRA_MACROS)))
YK_CXXFLAGS	+=	$(YK_CXXDBG) $(YK_CXXOPT) $(OMPFLAGS) $(YK_MACROS) $(YK_CXXWARN) $(EXTRA_YK_CXXFLAGS)
YK_LIBS		+=	$(EXTRA_YK_LIBS)

# Number of stencil/var dims extracted from YASK compiler output.
NSDIMS		:=	`awk '/define.*NUM_STENCIL_DIMS/ {print $$NF}' $(YK_CODE_FILE)`
NVDIMS		:=	`awk '/define.*NUM_STENCIL_AND_VAR_DIMS/ {print $$NF}' $(YK_CODE_FILE)`

######## Loop-compiler inputs:
# The loop indices range from 0..N-1.
# Dim 0 is the step dim, usually time.
# The step loop is handled outside of the generated loops,
# so the following loop codes do not scan over dim 0.
# Dims 1..N-1 are the domain dims, usually spatial.
# Thus, N-1 is the inner-most dim.
# For best perf, this should be the unit-stride dim in the vars.

# Rank loops break up the whole rank into smaller mega-blocks.  In order for
# temporal wavefronts to operate properly, the scanning paths must have
# strictly incrementing indices in each dim. Those that do not (e.g.,
# serpentine, square-wave) may *not* be used here when using temporal
# wavefronts. The time loop may be found in
# StencilEquations::run_solution(). The mega_blocks are executed sequentially,
# so no OpenMP here.
RANK_LOOP_MODS		?=	nested
RANK_LOOP_FLAGS		?=	-prefix rank_
RANK_LOOP_ORDER		?=	DOMAIN_LOOP_DIMS
RANK_LOOP_CODE		?=	$(RANK_LOOP_MODS) loop($(RANK_LOOP_ORDER)) { }

# Mega_Block loops break up a mega-block using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.
MEGA_BLOCK_LOOP_MODS	?=
MEGA_BLOCK_LOOP_OMP	?=	omp parallel for schedule(dynamic,1) proc_bind(spread)
MEGA_BLOCK_LOOP_FLAGS	?=	-prefix mega_block_ -omp '$(MEGA_BLOCK_LOOP_OMP)'
MEGA_BLOCK_LOOP_ORDER	?=	DOMAIN_LOOP_DIMS
MEGA_BLOCK_LOOP_CODE	?=	$(MEGA_BLOCK_LOOP_MODS) omp loop($(MEGA_BLOCK_LOOP_ORDER)) { } 

# Block loops break up a block into micro-blocks.
# These are executed sequentially, so no OpenMP.
# Like with rank loops, in order for temporal wavefronts to operate properly,
# the scanning paths must have strictly incrementing indices.
BLOCK_LOOP_MODS		?=	nested
BLOCK_LOOP_FLAGS	?=	-prefix block_
BLOCK_LOOP_ORDER	?=	DOMAIN_LOOP_DIMS
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_MODS) loop($(BLOCK_LOOP_ORDER)) { }

# Mini-Block loops break up a micro-block into nano-blocks. The 'omp' modifier
# creates a nested OpenMP loop so that each nano-block is assigned to a
# nested OpenMP thread. The OpenMP construct is not used when running with
# '-bind_inner_threads' because another parallel section is created.
MICRO_BLOCK_LOOP_MODS	?=
MICRO_BLOCK_LOOP_OMP	?=	omp parallel for schedule(static,1) proc_bind(spread)
MICRO_BLOCK_LOOP_FLAGS	?=	-prefix micro_block_ -omp '$(MICRO_BLOCK_LOOP_OMP)'
MICRO_BLOCK_LOOP_ORDER	?=	DOMAIN_LOOP_DIMS
MICRO_BLOCK_LOOP_CODE	?=	$(MICRO_BLOCK_LOOP_MODS) omp loop($(MICRO_BLOCK_LOOP_ORDER)) { }

# Nano-block loops break up a nano-block into pico-blocks.  These
# loops are run by a single OMP thread. When offloading, this loop is run in
# parallel on the offload device.  The indexes in this loop
# are 'normalized', i.e., vector units and rank-relative.
# When offloading, a nano-block is executed on the device.
NANO_BLOCK_LOOP_MODS	?=	serpentine
NANO_BLOCK_LOOP_FLAGS	?=	-no-align -prefix nano_block_
NANO_BLOCK_LOOP_ORDER	?=	DOMAIN_LOOP_DIMS
ifeq ($(offload),1)
NANO_BLOCK_LOOP_OMP	?=	omp target teams distribute thread_limit(thread_limit) device(KernelEnv::_omp_devn)
NANO_BLOCK_LOOP_FLAGS	+=	-omp '$(NANO_BLOCK_LOOP_OMP)'
NANO_BLOCK_LOOP_CODE	:=	$(NANO_BLOCK_LOOP_MODS) omp loop($(NANO_BLOCK_LOOP_ORDER)) { }
else
NANO_BLOCK_LOOP_CODE	:=	$(NANO_BLOCK_LOOP_MODS) loop($(NANO_BLOCK_LOOP_ORDER)) { }
endif

# Pico-Block loops break up a pico-block into vectors or clusters.
# When offloading, each pico-block is executed by a team of threads on the device.
# Pico-blocks must contain a separate inner loop over PICO_BLOCK_INNER_LOOP_DIM.
PICO_BLOCK_LOOP_FLAGS	?=	-no-align -prefix pico_block_
PICO_BLOCK_LOOP_ORDER	?=	PICO_BLOCK_OUTER_LOOP_DIMS
ifeq ($(offload),1)
PICO_BLOCK_LOOP_MODS	?=	omp nested
PICO_BLOCK_LOOP_OMP	?=	omp parallel for
PICO_BLOCK_LOOP_FLAGS	+=	-omp '$(PICO_BLOCK_LOOP_OMP)'
else
PICO_BLOCK_LOOP_MODS	?=	nested
endif
PICO_BLOCK_LOOP_CODE	?=	$(PICO_BLOCK_LOOP_MODS) loop($(PICO_BLOCK_LOOP_ORDER)) { \
				  loop(PICO_BLOCK_INNER_LOOP_DIM) { } }

# General-purpose parallel loop.
MISC_LOOP_OMP		?=	omp parallel for schedule(guided) proc_bind(spread)
MISC_LOOP_FLAGS		?=	-prefix misc_ -omp '$(MISC_LOOP_OMP)'
MISC_LOOP_ORDER		?=	DOMAIN_LOOP_DIMS
MISC_LOOP_CODE		?=	$(MISC_LOOP_MODS) omp loop($(MISC_LOOP_ORDER)) { }

# Source dirs.
VPATH	:=	$(patsubst $space,$colon,$(COMM_DIR) $(COEFF_DIR) $(YK_LIB_SRC_DIR))

# Compile rules.
$(YK_OBJ_DIR)/%.o: %.cpp $(YK_INC_GLOB)
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_EXT_OBJ_DIR)/%.o: %.cpp $(YK_GEN_HEADERS) $(YK_INC_GLOB)
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

# C++-preprocessor rule.
%.i: %.cpp $(YK_GEN_HEADERS) $(YK_INC_GLOB)
	g++ -E -CC $(YK_CXX_INCFLAGS) $< > $@
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
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $(YK_SO_FLAGS) -o $@ $^ $(YK_LIBS)
	@ls -l $@

$(YK_EXEC): yask_main.cpp $(YK_LIB)
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

$(YK_SCRIPT): ./yask.sh
	$(call MK_DIR,$(dir $@))
	cp $< $@
	chmod a+rx $@
	@ls -l $@

scripts: $(YK_SCRIPT)

$(MAKE_REPORT_FILE): $(YK_LIB)
	$(call MK_DIR,$(dir $@))
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	- $(MAKE) echo-settings >> $@ 2>&1

# Generated source files.
$(YK_GEN_DIR)/yask_rank_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(RANK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_RANK_LOOP_FLAGS) "$(RANK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_mega_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(MEGA_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_MEGA_BLOCK_LOOP_FLAGS) "$(MEGA_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_BLOCK_LOOP_FLAGS) "$(BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_micro_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(MICRO_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_MICRO_BLOCK_LOOP_FLAGS) "$(MICRO_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_nano_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(NANO_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_NANO_BLOCK_LOOP_FLAGS) "$(NANO_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_pico_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(PICO_BLOCK_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_PICO_BLOCK_LOOP_FLAGS) "$(PICO_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_misc_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -macro_file $(YK_CODE_FILE) -output $@ $(MISC_LOOP_FLAGS) $(EXTRA_LOOP_FLAGS) $(EXTRA_MISC_LOOP_FLAGS) "$(MISC_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -m $(NVDIMS) > $@
	@- $(INDENT) $@

$(YK_GEN_DIR)/yask_layouts.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -d $(NVDIMS) > $@
	@- $(INDENT) $@

$(YK_GEN_DIR)/yask_var_code.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(call MK_DIR,$(dir $@))
	$(PERL) $< -v $(NVDIMS) > $@

# Run the YASK compiler to create the stencil code.
# NB: '$(BASH) -o pipefail' ensures failure of command before '| tee' is visible.
$(YK_CODE_FILE): $(YC_EXEC)
	$(call MK_DIR,$(dir $@))
	@- rm -f $@
	$(BASH) -o pipefail -c \
	  "$(RUN_PREFIX) $< $(YC_FLAGS) -p $@ $(EXTRA_YC_FLAGS) 2>&1 | tee $(YC_REPORT_FILE)"
	@- $(INDENT) $@

# Pre-process the stencil code to create the context code.
# This adds the nano- and pico-loop code to make it easier for humans,
# debug tools, and performance tools to understand.
$(YK_SOLUTION_FILE): $(YK_CODE_FILE) $(YK_LOOP_FILES)
	$(BASH) -o pipefail -c \
	  "g++ -E -CC $(YK_CXX_INCFLAGS) $(YK_MACROS) -DDEFINE_SOLUTION $(YK_CODE_FILE) \
	    | $(PERL) -n -e '\$$start=1 if /Automatically-generated/; print if \$$start && !/^#/' > $@"
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
	$(call MK_DIR,$(dir $@))
	$(call MK_DIR,$(PY_OUT_DIR))
	$(SWIG) -v -DYK_MODULE=$(YK_PY_MOD_BASE) -cppext cpp \
	  -DUSE_MPI -DMPI_VERSION=3 -DMPI_Comm=int \
	  -I$(INC_DIR) -I$(COMM_DIR) -I$(COMM_DIR)/swig -I$(COEFF_DIR) \
	  -c++ -python -o $@ -outdir $(PY_OUT_DIR) -builtin $<
	$(SWIG_PATCH) $<

$(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o: $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.cpp
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ $(SWIG_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<
	@ls -l $@

$(YK_PY_LIB): $(YK_OBJS)  $(YK_EXT_OBJS) $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o
	$(call MK_DIR,$(dir $@))
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
	@echo "Build environment, `date`"; \
	echo TARGET=$(TARGET); \
	echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""; \
	echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""; \
	echo EXTRA_YK_CXXFLAGS="\"$(EXTRA_YK_CXXFLAGS)\""; \
	echo ISA=$(ISA); \
	echo MACROS="\"$(MACROS)\""; \
	echo OMPFLAGS="\"$(OMPFLAGS)\""; \
	echo VEC_MACROS="\"$(VEC_MACROS)\""; \
	echo YASK_OUTPUT_DIR=$(YASK_OUTPUT_DIR); \
	echo YC_FLAGS="\"$(YC_FLAGS)\""; \
	echo YC_STENCIL=$(YC_STENCIL); \
	echo YC_TARGET=$(YC_TARGET); \
	echo YK_ARCH=$(YK_ARCH); \
	echo YK_BASE=$(YK_BASE); \
	echo YK_EXEC=$(YK_EXEC); \
	echo YK_CXX=$(YK_CXX); \
	echo YK_CXXCMD=$(YK_CXXCMD); \
	echo YK_CXXDBG=$(YK_CXXDBG); \
	echo YK_CXXFLAGS="\"$(YK_CXXFLAGS)\""; \
	echo YK_CXXOPT=$(YK_CXXOPT); \
	echo YK_EXT_TAG=$(YK_EXT_TAG); \
	echo YK_STENCIL=$(YK_STENCIL); \
	echo YK_TAG=$(YK_TAG); \
	echo YK_CXXVER=`$(YK_CXX) --version`; \
	echo arch=$(arch); \
	echo cluster=$(cluster); \
	echo fold=$(fold); \
	echo offload=$(offload); \
	echo offload_usm=$(offload_usm); \
	echo pfd_l1=$(pfd_l1); \
	echo pfd_l2=$(pfd_l2); \
	echo radius=$(radius); \
	echo real_bytes=$(real_bytes); \
	echo stencil=$(stencil); \
	echo streaming_stores=$(streaming_stores)
	@echo " "; \
	echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""; \
	echo RANK_LOOP_FLAGS="\"$(RANK_LOOP_FLAGS)\""; \
	echo RANK_LOOP_ORDER="\"$(RANK_LOOP_ORDER)\""; \
	echo RANK_LOOP_MODS="\"$(RANK_LOOP_MODS)\""; \
	echo MEGA_BLOCK_LOOP_CODE="\"$(MEGA_BLOCK_LOOP_CODE)\""; \
	echo MEGA_BLOCK_LOOP_FLAGS="\"$(MEGA_BLOCK_LOOP_FLAGS)\""; \
	echo MEGA_BLOCK_LOOP_ORDER="\"$(MEGA_BLOCK_LOOP_ORDER)\""; \
	echo MEGA_BLOCK_LOOP_MODS="\"$(MEGA_BLOCK_LOOP_MODS)\""; \
	echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""; \
	echo BLOCK_LOOP_FLAGS="\"$(BLOCK_LOOP_FLAGS)\""; \
	echo BLOCK_LOOP_ORDER="\"$(BLOCK_LOOP_ORDER)\""; \
	echo BLOCK_LOOP_MODS="\"$(BLOCK_LOOP_MODS)\""; \
	echo MICRO_BLOCK_LOOP_CODE="\"$(MICRO_BLOCK_LOOP_CODE)\""; \
	echo MICRO_BLOCK_LOOP_FLAGS="\"$(MICRO_BLOCK_LOOP_FLAGS)\""; \
	echo MICRO_BLOCK_LOOP_ORDER="\"$(MICRO_BLOCK_LOOP_ORDER)\""; \
	echo MICRO_BLOCK_LOOP_MODS="\"$(MICRO_BLOCK_LOOP_MODS)\""; \
	echo NANO_BLOCK_LOOP_CODE="\"$(NANO_BLOCK_LOOP_CODE)\""; \
	echo NANO_BLOCK_LOOP_FLAGS="\"$(NANO_BLOCK_LOOP_FLAGS)\""; \
	echo NANO_BLOCK_LOOP_ORDER="\"$(NANO_BLOCK_LOOP_ORDER)\""; \
	echo NANO_BLOCK_LOOP_MODS="\"$(NANO_BLOCK_LOOP_MODS)\""; \
	echo PICO_BLOCK_LOOP_CODE="\"$(PICO_BLOCK_LOOP_CODE)\""; \
	echo PICO_BLOCK_LOOP_FLAGS="\"$(PICO_BLOCK_LOOP_FLAGS)\""; \
	echo PICO_BLOCK_LOOP_ORDER="\"$(PICO_BLOCK_LOOP_ORDER)\""; \
	echo PICO_BLOCK_LOOP_MODS="\"$(PICO_BLOCK_LOOP_MODS)\""; \
	echo MISC_LOOP_CODE="\"$(MISC_LOOP_CODE)\""; \
	echo MISC_LOOP_FLAGS="\"$(MISC_LOOP_FLAGS)\""; \
	echo MISC_LOOP_ORDER="\"$(MISC_LOOP_ORDER)\""; \
	echo MISC_LOOP_MODS="\"$(MISC_LOOP_MODS)\""; \
	git status
	uname -a

# Print stats on inner SIMD loops from asm file.
code-stats: $(YK_LIB)
	$(call MK_DIR,$(YK_OBJ_DIR))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) -x c++ -S -o $(YK_OBJ_DIR)/factory.s $(YK_LIB_SRC_DIR)/factory.cpp
	@echo "Code stats for stencil computation:"
	$(PERL) $(VIEW_ASM) -p -l -f='calc_' $(YK_OBJ_DIR)/factory.s

# Print some usage info.
help:
	@echo "Example builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd           # Build ISO3DFD stencil with default settings"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg               # Build SSG stencil with default settings"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg offload=1     # Device offload support"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg offload_usm=1 # Device offload support with unified shared memory"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg vtune=1       # Add VTune start/stop instrumentation"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg mpi=0         # No MPI support"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd radius=4  # Specify stencil radius"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd radius=2 YK_STENCIL=iso3d_r2 # Specify binary stencil name"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd arch=avx512    # Specify arch target"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg fold='x=1,y=2,z=4' # Specify folding (data layout)"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg cluster='x=2' # Specify clustering (unrolling)"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg real_bytes=8  # Use 8-byte (double) floats"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg YK_CXXOPT='-O2' # Use O2 optimization"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg YK_CXX=icpc   # Use classic Intel compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg YK_CXX=g++    # Use gnu compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=ssg MPI_CXX=mpiCC # Specify MPI compiler"; \
	echo " "
	@echo "Example builds of kernel API for C++ and Python apps:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd yk-api"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp yk-api"; \
	echo " "
	@echo "Example debug builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd mpi=0 omp=0 YK_CXXOPT='-O0' check=1 # No optimization, OpenMP, or MPI; use internal checking"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=avx2 stencil=test_2d YK_CXXOPT='-O0' trace=1   # Enable tracing; run with '-trace' to get trace"; \
	#echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=ssg radius=0 fold='x=1,y=1,z=1' YK_CXXOPT='-O0' trace_mem=1 # Trace all mem accesses with '-trace'"; \
	echo " "
	@echo "Example regression tests (run before any git push or pull request):"; \
	echo " $(MAKE) -j all                           # Normal full API and stencil tests"; \
	echo " $(MAKE) -j all offload=1                 # Test device offload"; \
	echo " $(MAKE) -j all offload_usm=1             # Test device offload with unified shared memory"; \
	echo " $(MAKE) -j all ranks=8                   # Test with 8 MPI ranks"; \
	echo " $(MAKE) -j all YK_CXXOPT=-O1             # Test at O1 optimization"; \
	echo " $(MAKE) -j all YK_CXX=g++                # Test w/g++"; \
	echo " $(MAKE) -j all mpi=0                     # Test w/o MPI"; \
	echo " $(MAKE) -j all YASK_OUTPUT_DIR=test-dir  # Test using alternate output dir"


#################################
########### Tests ###############
#################################
# TODO: convert all testing to a separate test framework.

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
ifeq ($(offload),1)
outer_threads	:=	2
inner_threads	:=	2
else
outer_threads	:=	8
inner_threads	:=	2
endif

# Makefile functions for folding and clustering.
# Disable folding and clustering for offload testing.
ifeq ($(offload),1)
FOLD		=
CLUSTER		=
else
FOLD		=	fold=$(subst $(space),$(comma),$(1))
CLUSTER		=	cluster=$(subst $(space),$(comma),$(1))
endif

TEST_MAKE_ARGS	:=	real_bytes=8 allow_new_var_types=0 check=1 trace=1
TEST_MAKE 	:=	$(MAKE) $(TEST_MAKE_ARGS)

### Unit tests.

$(YK_OMP_TEST_EXEC): $(YK_TEST_SRC_DIR)/openmp_test.cpp
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< -o $@ $(YK_LIBS)
	@ls -l $@

cxx-yk-omp-test:
	rm -f $(YK_OMP_TEST_EXEC)
	$(TEST_MAKE) $(YK_OMP_TEST_EXEC)
	@echo '*** Running the C++ YASK OpenMP test...'
	$(RUN_PREFIX) $(YK_OMP_TEST_EXEC)

$(YK_VAR_TEST_EXEC): $(YK_TEST_SRC_DIR)/var_test.cpp $(YK_LIB)
	$(call MK_DIR,$(dir $@))
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
	$(call MK_DIR,$(dir $@))
	$(CXX_PREFIX) $(YK_CXXCMD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

# Build and run C++ tests.
# First run on 1 rank, then multiple ranks if ranks>1.
cxx-yk-api-test:
	$(TEST_MAKE) $(YK_API_TEST_EXEC) allow_new_var_types=1
	@echo '*** Running the C++ YASK kernel API test...'
	$(RUN_PREFIX) $(YK_API_TEST_EXEC)
	if (( $(ranks) > 1 )); then $(RUN_PREFIX) mpirun -np $(ranks) $(YK_API_TEST_EXEC); fi

# Run Python kernel API test.
# First run on 1 rank, then multiple ranks if ranks>1.
py-yk-api-test: $(YK_TEST_SRC_DIR)/yask_kernel_api_test.py $(YK_PY_LIB)
	@echo '*** Running the Python YASK kernel API test...'
	$(RUN_PYTHON) $<
	if (( $(ranks) > 1 )); then mpirun -np $(ranks) $(RUN_PYTHON) $<; fi

# Build C++ kernel tests with exceptions.
$(YK_API_TEST_EXEC_WITH_EXCEPTION): $(YK_TEST_SRC_DIR)/yask_kernel_api_exception_test.cpp $(YK_LIB)
	$(call MK_DIR,$(dir $@))
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
	$(call MK_DIR,$(YK_GEN_DIR))
	mv $(YC_SRC_DIR)/yc-api-test-py.hpp $(YK_CODE_FILE)

# Run C++ compiler API test to create stencil-code file.
cxx-yc-api-test:
	$(MAKE) -C $(YC_SRC_DIR) $@
	$(call MK_DIR,$(YK_GEN_DIR))
	mv $(YC_SRC_DIR)/yc-api-test-cxx.hpp $(YK_CODE_FILE)

##### Validation runs for each binary.
DEF_TEST_ARGS := -log_dir $(TEST_LOG_OUT_DIR) \
	-ranks $(ranks) -stencil $(YK_STENCIL) -arch $(YK_ARCH) \
	-validate -no-pre_auto_tune -no-auto_tune -no-warmup -num_trials 1 -trial_steps 2 \
	-outer_threads $(outer_threads) -inner_threads $(inner_threads) \
	-ep 0 -mp 0 -no-allow_addl_padding
DEF_MPI_TEST_ARGS := $(DEF_TEST_ARGS) -min_exterior 0

# Test args for 1 and >1 ranks.
test_args0	:=	$(DEF_TEST_ARGS) -l 48 -Mb 32 -b 24 -Mbt 0 $(EXTRA_TEST_ARGS)
test_args1	:=	$(DEF_TEST_ARGS) -l 64 -b 48 -mb 32 -nb 24 -pb 16 -Mbt 1 $(EXTRA_TEST_ARGS)
test_args2	:=	$(DEF_TEST_ARGS) -l 63 -Mb 32 -b 24 -mb 16 -Mbt 2 -bundle_allocs $(EXTRA_TEST_ARGS)
test_args3	:=	$(DEF_TEST_ARGS) -l 63 -Mb 48 -b 24 -mb 16 -bt 2 -no-bundle_allocs $(EXTRA_TEST_ARGS)
ifeq ($(offload),1)
test_args4	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 $(EXTRA_TEST_ARGS)
else
test_args4	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -bind_inner_threads $(EXTRA_TEST_ARGS)
endif

# Test args for >1 ranks only.
ifeq ($(offload),1)
test_args10	:=	$(DEF_MPI_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -use_device_mpi -no-overlap_comms $(EXTRA_TEST_ARGS)
test_args11	:=	$(DEF_MPI_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -no-use_device_mpi -overlap_comms $(EXTRA_TEST_ARGS)
else
test_args10	:=	$(DEF_MPI_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -no-use_shm -overlap_comms $(EXTRA_TEST_ARGS)
test_args11	:=	$(DEF_MPI_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -use_shm -no-overlap_comms $(EXTRA_TEST_ARGS)
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
# This is the primary target for building and running stencil tests.
yc-and-yk-test: $(YK_EXEC) $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests yk-mpi-tests; fi

# Run the YASK kernel test without implicity using the YASK compiler.
yk-test-no-yc: kernel-no-yc $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests yk-mpi-tests; fi

# Run the kernel API tests for C++ and Python with and w/o expected exceptions.
api-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-api-test
	$(MAKE) clean; $(MAKE) py-yk-api-test
	$(MAKE) clean; $(MAKE) cxx-yk-api-test-with-exception
	$(MAKE) clean; $(MAKE) py-yk-api-test-with-exception

# Run several stencils using built-in validation.
# NB: set arch var as applicable.
# NB: save some compile time by using YK_CXXOPT=-O2 or -O1.
# These tests are focused on the kernel and not the compiler.
# For testing both the kernel and compiler in various combinations,
# run the tests from the top-level Makefile.
STENCIL_TEST	:=	$(TEST_MAKE) yc-and-yk-test

1d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stream_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=4) $(call CLUSTER,x=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d YK_STENCIL_SUFFIX=-t3 $(call FOLD,x=4) $(call CLUSTER,x=5)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_1d YK_STENCIL_SUFFIX=-t4 $(call FOLD,x=4) $(call CLUSTER,x=5) read_ahead_dist=3
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_func_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_step_cond_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_1d $(call FOLD,x=4)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_boundary_1d $(call FOLD,x=4)

2d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_empty step_dim=t domain_dims=d1,d2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_empty_2d
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=2 y=2) inner_loop_dim=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=2 y=2) inner_loop_dim=2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t3 $(call FOLD,x=2 y=2) $(call CLUSTER,y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t4 $(call FOLD,x=2 y=2) $(call CLUSTER,x=3 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t5 $(call FOLD,x=2 y=2) mpi=0 ranks=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_2d YK_STENCIL_SUFFIX=-t6 arch=intel64 $(call FOLD,x=3 y=2) domain_dims=y,x
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_reverse_2d YK_STENCIL_SUFFIX=-t1 radius=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_reverse_2d YK_STENCIL_SUFFIX=-t2 radius=1 read_ahead_dist=2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_2d $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=2 y=2) inner_misc_layout=0 outer_domain_layout=0
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=2 y=2) inner_misc_layout=0 outer_domain_layout=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d YK_STENCIL_SUFFIX=-t3 $(call FOLD,x=2 y=2) inner_misc_layout=1 outer_domain_layout=0
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_misc_2d YK_STENCIL_SUFFIX=-t4 $(call FOLD,x=2 y=2) inner_misc_layout=1 outer_domain_layout=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_2d $(call FOLD,x=2 y=2) read_ahead_dist=2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_2d $(call FOLD,x=2 y=2)

3d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stream_3d radius=5
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=2 y=2 z=2) EXTRA_YC_FLAGS=-first-inner
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=2 y=2 z=2) EXTRA_YC_FLAGS=-no-first-inner
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d YK_STENCIL_SUFFIX=-t3 $(call FOLD,x=2 z=2) domain_dims=z,y,x
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d YK_STENCIL_SUFFIX=-t4 $(call FOLD,x=2 z=2) inner_loop_dim=2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_3d YK_STENCIL_SUFFIX=-t5 $(call FOLD,x=2 z=2) NANO_BLOCK_LOOP_MODS=serpentine
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_stages_3d $(call FOLD,y=2 x=2) domain_dims=x,z,y
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_partial_3d YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=2 z=2) read_ahead_dist=2
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_partial_3d YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=2 z=2) inner_loop_dim=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_partial_3d YK_STENCIL_SUFFIX=-t3 $(call FOLD,x=2 z=2) domain_dims=x,z,y
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_boundary_3d $(call FOLD,x=2 y=2) inner_loop_dim=1
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_scratch_3d $(call FOLD,x=2 z=2) inner_loop_dim=x

# 3D tests w/specific shapes.
3d-tests2:
	$(MAKE) clean; $(STENCIL_TEST) stencil=3axis YK_STENCIL_SUFFIX=-t1 $(call FOLD,x=2 y=2) $(call CLUSTER,x=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=3axis YK_STENCIL_SUFFIX=-t2 $(call FOLD,x=2 y=2) $(call CLUSTER,z=2 y=2)
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
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg_merged $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg_merged_abc $(call FOLD,x=2 y=2)

# 3D tests w/seismic stencils that may generate too many kernel parameters for offload.
3d-tests4:
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg $(call FOLD,x=2 y=2)
	$(MAKE) clean; $(STENCIL_TEST) stencil=fsg_abc $(call FOLD,x=2 y=2)

4d-tests:
	$(MAKE) clean; $(STENCIL_TEST) stencil=test_4d $(call FOLD,w=2 x=2)

stencil-tests:
	$(MAKE) 1d-tests
	$(MAKE) 2d-tests
	$(MAKE) 3d-tests
	$(MAKE) 3d-tests2
	$(MAKE) 3d-tests3
	if (( $(offload) == 0 )); then $(MAKE) 3d-tests4; fi
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

