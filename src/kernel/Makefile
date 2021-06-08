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
omp_par			?=	omp parallel
omp_par_for		?=	omp parallel for
omp_for			?=	omp for
omp_region_schedule	?=	dynamic,1
omp_block_schedule	?=	static,1
omp_misc_schedule	?=	guided
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
# The "knc" arch must be set manually because it is always cross-compiled.
# (Do not assume 'yask.sh' has been installed in $(BIN_OUT_DIR) yet.)
arch			?=	$(shell $(BASH) ./yask.sh -show_arch)

# Set 'TARGET' from 'arch', converting codenames and other aliases to ISA names.
# 'TARGET' is the canonical target name.
# The possible values must agree with those in the APIs and YASK compiler.
ifneq ($(filter $(arch),avx snb ivb),)
  TARGET		:=	avx
else ifneq ($(filter $(arch),avx2 hsw bdw),)
  TARGET		:=	avx2
else ifneq ($(filter $(arch),avx512lo),)
  TARGET		:=	avx512lo
else ifneq ($(filter $(arch),avx512 avx512hi avx512f skx skl clx),)
  TARGET		:=	avx512
else ifneq ($(filter $(arch),knl),)
  TARGET		:=	knl
else ifneq ($(filter $(arch),knc),)
  TARGET		:=	knc
else ifneq ($(filter $(arch),intel64 cpp),)
  TARGET		:=	intel64
else
  $(error Target not recognized; use arch=avx512, avx512lo, avx2, avx, knl, knc, or intel64)
endif

# Specify YK_STENCIL and/or YK_ARCH on 'make' cmd-line to name binaries differently.
YK_STENCIL	:=	$(stencil)
YK_ARCH		:=	$(arch)

# Specify YC_STENCIL on 'make' cmd-line to pass different stencil name to YASK compiler.
YC_STENCIL	:=	$(stencil)

# Specify YC_TARGET on 'make' cmd-line to override target for YASK compiler.
YC_TARGET	:=	$(TARGET)

# Default compiler settings based on target.
ifeq ($(TARGET),knc)

 ICC_ISA	?=	-mmic
 MACROS		+=	USE_INTRIN512

else ifeq ($(TARGET),knl)

 # NUMA_PREF=1 causes YASK to use MCDRAM on a node in "flat" memory mode. It
 # will fall-back to NUMA node 0 in "cache" memory mode. It will *NOT* do
 # the right thing by default in SNC-2 or SNC-4 clustering modes (using
 # either flat or cache memory modes, but these configurations are uncommon.
 ICC_ISA	?=	-xMIC-AVX512
 CXX_ISA	?=	-march=knl
 MACROS		+=	USE_INTRIN512 USE_RCP28 NUMA_PREF=1

else ifeq ($(TARGET),avx512)

 ICC_ISA	?=	-xCORE-AVX512
 CXX_ISA	?=	-march=skylake-avx512
 MACROS		+=	USE_INTRIN512

else ifeq ($(TARGET),avx512lo)

 ICC_ISA	?=	-xCORE-AVX512
 CXX_ISA	?=	-march=skylake-avx512
 MACROS		+=	USE_INTRIN512LO

else ifeq ($(TARGET),avx2)

 ICC_ISA	?=	-xCORE-AVX2
 CXX_ISA	?=	-march=haswell
 MACROS		+=	USE_INTRIN256 USE_AVX2

else ifeq ($(TARGET),avx)

 ICC_ISA	?=	-xAVX
 CXX_ISA	?=	-march=sandybridge
 MACROS		+=	USE_INTRIN256

else ifeq ($(TARGET),intel64)

 ICC_ISA	?=	-xHOST
 CXX_ISA	?=	-march=native

else

$(error Internal error: arch lists are not consistent)

endif # arch-specific.

# Run knobs
ifeq ($(mpi),0)
 YK_SCRIPT_OPTS	+= -ranks 1
endif
YK_SCRIPT_OPTS	+= "-stencil" $(YK_STENCIL) "-arch" $(YK_ARCH)

# Turn off prefetching at O0 or O1.
ifneq ($(filter -O0 -O1,$(YK_CXXOPT)),)
 pfd_l1		:=	0
 pfd_l2		:=	0
endif

# Turn on checking at O0.
ifneq ($(filter -O0,$(YK_CXXOPT)),)
 check		:=	1
endif

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# The remainder of this file specifies how to build and test the kernel.
# Vars starting with 'YC_' apply to the stencil compiler.
# Vars starting with 'YK_' apply to the stencil kernel.

# Flags passed to stencil compiler.
YC_FLAGS	:=	-stencil $(YC_STENCIL) -target $(YC_TARGET) -elem-bytes $(real_bytes)
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
YK_TAG		:=	$(YK_STENCIL).$(YK_ARCH)
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
YK_SRC_NAMES	:=	utils
YK_EXT_SRC_NAMES :=	factory soln_apis context stencil_calc setup alloc \
			generic_var yk_var yk_var_apis new_var settings auto_tuner
YK_OBJS		:=	$(addprefix $(YK_OBJ_DIR)/,$(addsuffix .o,$(YK_SRC_NAMES) $(COMM_SRC_NAMES)))
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
YK_VAR_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_var_test.exe
YK_API_TEST_EXEC_WITH_EXCEPTION :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_exception_test.exe
MAKE_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).make-report.txt
YC_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).yask_compiler-report.txt

# File-related macros.
MACROS		+= 	YK_CODE_FILE=$(YK_CODE_FILE)

# YASK loop compiler.
GEN_LOOPS	:=	$(UTILS_BIN_DIR)/gen_loops.pl

# Other scripts and cmds.
GEN_LAYOUTS	:=	$(UTILS_BIN_DIR)/gen_layouts.pl
VIEW_ASM	:=	$(UTILS_BIN_DIR)/view_asm.pl

# Compiler and default flags.
ifeq ($(mpi),1)
 YK_CXX		:=	mpiicc
else
 YK_CXX		:=	icc
endif
YK_CXXFLAGS	:=	-g -std=c++11 -Wall
YK_CXXFLAGS	+=	$(addprefix -I,$(YK_INC_DIRS) $(YK_GEN_DIR))
OMPFLAGS	+=	-qopenmp
YK_CXXOPT	?=	-O3
SWIG_CXXFLAGS	:= 	$(DBL_EPSILON_CXXFLAG)

# Linker.
YK_LD		:=	$(YK_CXX)
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

endif

# Add options for PMEM.
ifeq ($(pmem),1)
 YK_CXXFLAGS	+=	-I/opt/intel/memkind/include
 YK_LIBS	+=	-L/opt/intel/memkind/lib -lmemkind
 MACROS		+=	USE_PMEM
endif

# Turn on debug macros.
ifeq ($(trace_mem),1)
 MACROS		+=	CHECK TRACE TRACE_MEM
else ifeq ($(trace),1)
 MACROS		+=	CHECK TRACE
else ifeq ($(check),1)
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

# MPI settings.
ifeq ($(mpi),1)
 MACROS		+=	USE_MPI
endif

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
 VTUNE_GENERIC_DIR	:=	/opt/intel/vtune_profiler
 YK_CXXFLAGS	+=	-I$(VTUNE_GENERIC_DIR)/include
 YK_LIBS	+=	-L$(VTUNE_GENERIC_DIR)/lib64
 ifneq ($(VTUNE_PROFILER_2020_DIR),)
  VTUNE_DIR	:=	$(VTUNE_PROFILER_2020_DIR)
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
 YK_VTUNE_CMD	:=	echo "*** VTune command:" vtune -c hpc-performance $(YK_SCRIPT) $(YK_SCRIPT_OPTS) "[options]"
endif

# compiler-specific settings.

# Clang (LLVM-based) compiler.
ifeq ($(call MACRO_DEF,$(YK_CXX) $(EXTRA_YK_CXXFLAGS),__clang__),1)
 YK_CXXFLAGS	+=	$(CXX_ISA)
 YK_CXXFLAGS	+=	-Wno-unused-variable 
 MACROS		+=	NO_VEC
 MACROS		+=	NO_SVML

# Intel compiler (and not clang-based).
else ifeq ($(call MACRO_DEF,$(YK_CXX) $(EXTRA_YK_CXXFLAGS),__INTEL_COMPILER),1)

 YK_CXXFLAGS	+=	$(ICC_ISA)
 YK_CXXFLAGS	+=	-debug extended -restrict -ansi-alias -fno-alias
 YK_CXXFLAGS	+=	-fimf-use-svml=true -fimf-precision=low -fast-transcendentals \
			-no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -ftz \
			-fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 YK_CXXFLAGS	+=	-no-diag-message-catalog
 YK_CXXFLAGS	+=	-vec-threshold0
 #YK_CXXFLAGS	+=	-qoverride-limits
 #YK_CXXFLAGS	+=	-Fa
 #YK_CXXFLAGS	+=	-qopt-report=5
 #YK_CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 YK_CXX_VER_CMD :=	$(YK_CXX) -V

 # Use real 512-bit code.
 # (Commented out because it doesn't help performance, and it breaks Devito.)
 ifeq ($(TARGET),avx512)
  #YK_CXXFLAGS	+=	-qopt-zmm-usage=high
 endif

# Gnu-compatible compiler (but not Intel or Clang).
else ifeq ($(call MACRO_DEF,$(YK_CXX) $(EXTRA_YK_CXXFLAGS),__GNUC__),1)
 YK_CXXFLAGS	+=	$(CXX_ISA)
 YK_CXXFLAGS	+=	-Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable
 SWIG_CXXFLAGS	+=	$(SWIG_GCCFLAGS)
 MACROS		+=	NO_SVML

# NB: if none of the above tests is triggered, the ISA will not be specified.

endif # compiler.

# Compile with model_cache=1 or 2 to check prefetching.
# Turn off OpenMP when modeling a cache.
# This is currently not operative! TODO: re-enable cache model.
ifeq ($(model_cache),1)
 MACROS		+=	MODEL_CACHE=1
 OMPFLAGS	:=	-qopenmp-stubs
else ifeq ($(model_cache),2)
 MACROS		+=	MODEL_CACHE=2
 OMPFLAGS	:=	-qopenmp-stubs
endif

# Add in final flags and user-added flags.
YK_CXXFLAGS	+=	$(YK_CXXOPT) $(OMPFLAGS) $(addprefix -D,$(MACROS) $(EXTRA_MACROS)) $(EXTRA_YK_CXXFLAGS)
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
# StencilEquations::run_solution().
RANK_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar rank_idxs
RANK_LOOP_ORDER		?=	1 .. N-1
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop($(RANK_LOOP_ORDER)) { \
				 $(RANK_LOOP_INNER_MODS) call(calc_region(bp)); }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.
REGION_LOOP_OPTS	?=	-ndims $(NSDIMS) -inVar region_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_region_schedule)) proc_bind(spread)'
REGION_LOOP_ORDER	?=	1 .. N-1
REGION_LOOP_CODE	?=	$(REGION_LOOP_OUTER_MODS) omp loop($(REGION_LOOP_ORDER)) { \
				 $(REGION_LOOP_INNER_MODS) \
				  call(calc_block(bp, region_shift_num, nphases, phase, rank_idxs)); }

# Block loops break up a block into mini-blocks.
# These are executed sequentially, so no OpenMP.
# Like with rank loops, in order for temporal wavefronts to operate properly,
# the scanning paths must have strictly incrementing indices. 
BLOCK_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar adj_block_idxs
BLOCK_LOOP_ORDER	?=	1 .. N-1
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_OUTER_MODS) loop($(BLOCK_LOOP_ORDER)) { \
				 $(BLOCK_LOOP_INNER_MODS) \
				  call(calc_mini_block(region_thread_idx, bp, region_shift_num, \
					nphases, phase, nshapes, shape, \
					bridge_mask, rank_idxs, region_idxs, block_idxs)); }

# Mini-Block loops break up a mini-block into sub-blocks. The 'omp' modifier
# creates a *nested* OpenMP loop (but not a parallem region) so that each
# sub-block is assigned to a nested OpenMP thread. The OpenMP construct is
# not used when running with '-bind_block_threads'.
MINI_BLOCK_LOOP_OPTS	?=	-ndims $(NSDIMS) -inVar adj_mb_idxs \
				-ompConstruct '$(omp_for) schedule($(omp_block_schedule)) nowait'
MINI_BLOCK_LOOP_ORDER	?=	1 .. N-1
MINI_BLOCK_LOOP_CODE	?=	$(MINI_BLOCK_LOOP_OUTER_MODS) omp loop($(MINI_BLOCK_LOOP_ORDER)) { \
				 $(MINI_BLOCK_LOOP_INNER_MODS) call(CALC_SUB_BLOCK); }

# Sub-block loops break up a sub-block into clusters or vectors.  These
# loops are run by a single OMP thread.  The N-1 (inner) loop is generated
# by the stencil compiler, so these loops stop at N-2.  The indexes in this
# loop are 'normalized', i.e., vector units and rank-relative.
SUB_BLOCK_LOOP_OPTS	?=	-ndims $(NSDIMS) -inVar norm_sub_block_idxs
SUB_BLOCK_LOOP_ORDER	?=	1 .. N-2
SUB_BLOCK_LOOP_CODE	?=	$(SUB_BLOCK_LOOP_OUTER_MODS) loop($(SUB_BLOCK_LOOP_ORDER)) { \
				 $(SUB_BLOCK_LOOP_INNER_MODS) call(CALC_INNER_LOOP); }

# General-purpose parallel loop.
MISC_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar misc_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_misc_schedule)) proc_bind(spread)'
MISC_LOOP_ORDER		?=	1 .. N-1
MISC_LOOP_CODE		?=	$(MISC_LOOP_OUTER_MODS) omp loop($(MISC_LOOP_ORDER)) { \
				$(MISC_LOOP_INNER_MODS) call(MISC_FN); }

# Compile rules.

$(YK_OBJ_DIR)/%.o: $(COMM_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_OBJ_DIR)/%.o: $(COEFF_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_OBJ_DIR)/%.o: $(YK_LIB_SRC_DIR)/%.cpp $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

$(YK_EXT_OBJ_DIR)/%.o: $(YK_LIB_SRC_DIR)/%.cpp $(YK_GEN_HEADERS) $(YK_INC_GLOB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -x c++ -fPIC -c -o $@ $<
	@ls -l $@

######## Primary targets.
# NB: must set stencil and arch make vars to generate the desired YASK kernel.

default: kernel

# Compile the kernel.
kernel: $(YK_EXEC) $(YK_SCRIPT) $(MAKE_REPORT_FILE)
	@echo "*** Binary" $(YK_EXEC) "has been built."; \
	echo "*** See" $(MAKE_REPORT_FILE) "for build information."; \
	echo "*** Run command:" $(YK_SCRIPT) $(YK_SCRIPT_OPTS) "[options]"
	@$(YK_VTUNE_CMD)

$(YK_LIB): $(YK_OBJS) $(YK_EXT_OBJS)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^ $(YK_LIBS)
	@ls -l $@

$(YK_EXEC): yask_main.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_LD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@ $(YK_LIBS)
	@ls -l $@

$(YK_SCRIPT): ./yask.sh
	$(MKDIR) $(dir $@)
	cp $< $@
	chmod a+rx $@
	@ls -l $@

$(MAKE_REPORT_FILE): $(YK_LIB)
	$(MKDIR) $(dir $@)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	- $(MAKE) echo-settings >> $@ 2>&1

# Generated source files.
$(YK_GEN_DIR)/yask_rank_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(RANK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_RANK_LOOP_OPTS) "$(RANK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_region_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

$(YK_GEN_DIR)/yask_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_mini_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(MINI_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_MINI_BLOCK_LOOP_OPTS) "$(MINI_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_sub_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(SUB_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_SUB_BLOCK_LOOP_OPTS) "$(SUB_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_misc_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(MISC_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_MISC_LOOP_OPTS) "$(MISC_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -m $(NVDIMS) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_layouts.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -d $(NVDIMS) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_var_code.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -v $(NVDIMS) > $@

# NB: '$(BASH) -o pipefail' ensures failure of command before '| tee' is visible.
$(YK_CODE_FILE): $(YC_EXEC)
	$(MKDIR) $(dir $@)
	@- rm -f $@
	$(BASH) -o pipefail -c \
	  "$(RUN_PREFIX) $< $(YC_FLAGS) -p $@ $(EXTRA_YC_FLAGS) 2>&1 | tee $(YC_REPORT_FILE)"
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because no indent program was found."

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
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -x c++ $(SWIG_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<
	@ls -l $@

$(YK_PY_LIB): $(YK_OBJS)  $(YK_EXT_OBJS) $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^ $(YK_LIBS)
	@ls -l $@

# Simple tests.

$(YK_VAR_TEST_EXEC): $(YK_TEST_SRC_DIR)/var_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@
	@ls -l $@

#### API tests.

# Build C++ kernel tests.
$(YK_API_TEST_EXEC): $(YK_TEST_SRC_DIR)/yask_kernel_api_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@
	@ls -l $@

# Run C++ tests.
cxx-yk-var-test: $(YK_VAR_TEST_EXEC)
	@echo '*** Running the C++ YASK var test...'
	$(RUN_PREFIX) $<

cxx-yk-api-test: $(YK_API_TEST_EXEC)
	@echo '*** Running the C++ YASK kernel API test...'
	$(RUN_PREFIX) $<

# Run Python kernel API test.
py-yk-api-test: $(YK_TEST_SRC_DIR)/yask_kernel_api_test.py $(YK_PY_LIB)
	@echo '*** Running the Python YASK kernel API test...'
	$(RUN_PYTHON) $<

# Build C++ kernel tests with exceptions.
$(YK_API_TEST_EXEC_WITH_EXCEPTION): $(YK_TEST_SRC_DIR)/yask_kernel_api_exception_test.cpp $(YK_LIB)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@
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

# Validation runs for each binary.
DEF_TEST_ARGS := -ranks $(ranks) -stencil $(YK_STENCIL) -arch $(YK_ARCH) -v \
	-trial_steps 2 -max_threads 6 -block_threads 2 \
	-ep 0 -mp 0 -no-allow_addl_padding

val0	:=	$(DEF_TEST_ARGS) -l 48 -r 32 -b 24 -rt 0 $(EXTRA_TEST_ARGS)
val1	:=	$(DEF_TEST_ARGS) -l 48 -r 32 -b 24 -mb 16 -rt 1 $(EXTRA_TEST_ARGS)
val2	:=	$(DEF_TEST_ARGS) -l 63 -r 32 -b 24 -mb 16 -rt 2 $(EXTRA_TEST_ARGS)
val3	:=	$(DEF_TEST_ARGS) -l 63 -r 48 -b 24 -mb 16 -bt 2 $(EXTRA_TEST_ARGS)
val4a	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -bind_block_threads $(EXTRA_TEST_ARGS)
val4b	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -use_shm -overlap_comms $(EXTRA_TEST_ARGS)
val4c	:=	$(DEF_TEST_ARGS) -l 48 -b 24 -mb 16 -bt 2 -use_shm -no-overlap_comms $(EXTRA_TEST_ARGS)
ranks	:=	2

# Run the kernel binary using several combos of sizes and ranks.
yk-tests:
	$(YK_SCRIPT) $(val0)
	$(YK_SCRIPT) $(val1)
	$(YK_SCRIPT) $(val2)
	$(YK_SCRIPT) $(val3)
	$(YK_SCRIPT) $(val4a)

yk-mpi-tests:
	$(YK_SCRIPT) $(val4b)
	$(YK_SCRIPT) $(val4c)

# Run the default YASK compiler and kernel.
yc-and-yk-test: $(YK_EXEC) $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests yk-mpi-tests; fi

# Run the YASK kernel test without implicity using the YASK compiler.
yk-test-no-yc: kernel-no-yc $(YK_SCRIPT)
	$(MAKE) ranks=1 yk-tests
	if (( $(ranks) > 1 )); then $(MAKE) yk-tests; fi

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
STENCIL_TEST_ARGS := yc-and-yk-test real_bytes=8 allow_new_var_types=0 check=1 trace=1
stencil-tests:
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_empty step_dim=t domain_dims=d1,d2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_empty_2d
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_2d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_2d fold=x=2,y=2 mpi=0 ranks=1
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_2d arch=intel64 fold=x=3,y=2 domain_dims=y,x
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_3d fold=x=2,z=2 domain_dims=z,y,x
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_partial_3d fold=x=2,z=2 domain_dims=x,z,y
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_stream_3d radius=5
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_reverse_2d radius=1
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_func_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_stages_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_stages_2d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_stages_3d fold=y=2,x=2 domain_dims=x,z,y
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_misc_2d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_misc_2d fold=x=2,y=2 EXTRA_YC_FLAGS=-interleave-misc
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_step_cond_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_boundary_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_boundary_2d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_boundary_3d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_scratch_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_scratch_3d fold=x=2,z=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_scratch_boundary_1d fold=x=4
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=iso3dfd radius=3 fold=x=2,y=2 domain_dims=z,x,y
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=iso3dfd_sponge radius=3 fold=x=2,z=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=3axis fold=x=2,y=2 cluster=x=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=3axis fold=x=2,y=2 cluster=z=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=3axis_with_diags fold=x=2,z=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=3plane fold=y=2,z=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=cube fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=ssg fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=ssg_merged fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=awp fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=awp_abc fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=fsg_abc fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=fsg_merged fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=tti fold=x=2,y=2
	$(MAKE) clean; $(MAKE) $(STENCIL_TEST_ARGS) stencil=test_4d fold=w=2,x=2

all-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-var-test stencil=test_3d fold=x=4,y=2
	$(MAKE) stencil-tests
	$(MAKE) api-tests

all:
	$(MAKE) kernel
	$(MAKE) api
	$(MAKE) all-tests

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
	rm -rf logs

echo-settings:
	@echo "Build environment for" $(YK_EXEC) on `date`; \
	echo TARGET=$(TARGET); \
	echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""; \
	echo BLOCK_LOOP_INNER_MODS="\"$(BLOCK_LOOP_INNER_MODS)\""; \
	echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""; \
	echo BLOCK_LOOP_ORDER="\"$(BLOCK_LOOP_ORDER)\""; \
	echo BLOCK_LOOP_OUTER_MODS="\"$(BLOCK_LOOP_OUTER_MODS)\""; \
	echo BLOCK_LOOP_OUTER_VARS="\"$(BLOCK_LOOP_OUTER_VARS)\""; \
	echo DEF_ARGS="\"$(DEF_ARGS)\""; \
	echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""; \
	echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""; \
	echo EXTRA_YK_CXXFLAGS="\"$(EXTRA_YK_CXXFLAGS)\""; \
	echo ISA=$(ISA); \
	echo MACROS="\"$(MACROS)\""; \
	echo MINI_BLOCK_LOOP_CODE="\"$(MINI_BLOCK_LOOP_CODE)\""; \
	echo MINI_BLOCK_LOOP_INNER_MODS="\"$(MINI_BLOCK_LOOP_INNER_MODS)\""; \
	echo MINI_BLOCK_LOOP_OPTS="\"$(MINI_BLOCK_LOOP_OPTS)\""; \
	echo MINI_BLOCK_LOOP_ORDER="\"$(MINI_BLOCK_LOOP_ORDER)\""; \
	echo MINI_BLOCK_LOOP_OUTER_MODS="\"$(MINI_BLOCK_LOOP_OUTER_MODS)\""; \
	echo MINI_BLOCK_LOOP_OUTER_VARS="\"$(MINI_BLOCK_LOOP_OUTER_VARS)\""; \
	echo MISC_LOOP_CODE="\"$(MISC_LOOP_CODE)\""; \
	echo MISC_LOOP_INNER_MODS="\"$(MISC_LOOP_INNER_MODS)\""; \
	echo MISC_LOOP_OPTS="\"$(MISC_LOOP_OPTS)\""; \
	echo MISC_LOOP_ORDER="\"$(MISC_LOOP_ORDER)\""; \
	echo MISC_LOOP_OUTER_MODS="\"$(MISC_LOOP_OUTER_MODS)\""; \
	echo MISC_LOOP_OUTER_VARS="\"$(MISC_LOOP_OUTER_VARS)\""; \
	echo OMPFLAGS="\"$(OMPFLAGS)\""; \
	echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""; \
	echo RANK_LOOP_INNER_MODS="\"$(RANK_LOOP_INNER_MODS)\""; \
	echo RANK_LOOP_OPTS="\"$(RANK_LOOP_OPTS)\""; \
	echo RANK_LOOP_ORDER="\"$(RANK_LOOP_ORDER)\""; \
	echo RANK_LOOP_OUTER_MODS="\"$(RANK_LOOP_OUTER_MODS)\""; \
	echo RANK_LOOP_OUTER_VARS="\"$(RANK_LOOP_OUTER_VARS)\""; \
	echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""; \
	echo REGION_LOOP_INNER_MODS="\"$(REGION_LOOP_INNER_MODS)\""; \
	echo REGION_LOOP_OPTS="\"$(REGION_LOOP_OPTS)\""; \
	echo REGION_LOOP_ORDER="\"$(REGION_LOOP_ORDER)\""; \
	echo REGION_LOOP_OUTER_MODS="\"$(REGION_LOOP_OUTER_MODS)\""; \
	echo REGION_LOOP_OUTER_VARS="\"$(REGION_LOOP_OUTER_VARS)\""; \
	echo SUB_BLOCK_LOOP_CODE="\"$(SUB_BLOCK_LOOP_CODE)\""; \
	echo SUB_BLOCK_LOOP_INNER_MODS="\"$(SUB_BLOCK_LOOP_INNER_MODS)\""; \
	echo SUB_BLOCK_LOOP_INNER_VARS="\"$(SUB_BLOCK_LOOP_INNER_VARS)\""; \
	echo SUB_BLOCK_LOOP_OPTS="\"$(SUB_BLOCK_LOOP_OPTS)\""; \
	echo SUB_BLOCK_LOOP_ORDER="\"$(SUB_BLOCK_LOOP_ORDER)\""; \
	echo SUB_BLOCK_LOOP_OUTER_MODS="\"$(SUB_BLOCK_LOOP_OUTER_MODS)\""; \
	echo SUB_BLOCK_LOOP_OUTER_VARS="\"$(SUB_BLOCK_LOOP_OUTER_VARS)\""; \
	echo YASK_OUTPUT_DIR=$(YASK_OUTPUT_DIR); \
	echo YC_FLAGS="\"$(YC_FLAGS)\""; \
	echo YC_STENCIL=$(YC_STENCIL); \
	echo YC_TARGET=$(YC_TARGET); \
	echo YK_ARCH=$(YK_ARCH); \
	echo YK_BASE=$(YK_BASE); \
	echo YK_CXX=$(YK_CXX); \
	echo YK_CXXFLAGS="\"$(YK_CXXFLAGS)\""; \
	echo YK_CXXOPT=$(YK_CXXOPT); \
	echo YK_EXT_TAG=$(YK_EXT_TAG); \
	echo YK_STENCIL=$(YK_STENCIL); \
	echo YK_TAG=$(YK_TAG); \
	echo arch=$(arch); \
	echo cluster=$(cluster); \
	echo fold=$(fold); \
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
# Compile with 'EXTRA_YK_CXXFLAGS=-Fa CXX_PREFIX=' to make asm file.
# (The 'CXX_PREFIX=' part just turns off 'ccmake' because it
# won't rebuild if only '-Fa' is added.)
code-stats: kernel
	@echo "Code stats for stencil computation:"
	$(PERL) $(VIEW_ASM) -p -l -f=calc_loop_of_clusters factory.s

# Print some usage info.
help:
	@echo "Example performance builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd arch=knl # Explicit arch target"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd radius=4 YK_STENCIL=iso3dfd_rt # Specify binary stencil name"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis fold='x=1,y=2,z=4' cluster='x=2'"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis radius=4 real_bytes=8 # DP FP"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp EXTRA_YK_CXXFLAGS='-qopt-report -Fa'"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis EXTRA_YK_CXXFLAGS='-qnextgen' # Use next-gen Intel compiler"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis YK_CXX=clang++ mpi=0 # Use clang w/o MPI"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis YK_CXX=clang++ EXTRA_YK_CXXFLAGS='-fno-color-diagnostics' mpi=0 # Use clang w/o color output"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=3axis YK_CXX=mpiCC OMPI_CXX=clang++ # Use clang w/MPI"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp vtune=1 # Add VTune start/stop instrumentation"; \
	echo " "
	@echo "Example performance builds of kernel API for C++ and Python apps:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd yk-api"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=awp yk-api"; \
	echo " "
	@echo "Example debug builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' check=1 # No optimization, OpenMP or MPI; internal checking"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=avx stencil=test_2d mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' trace=1 # Enable tracing; run with '-trace' to get trace"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' mpi=0 YK_CXX=g++ OMPFLAGS='' YK_CXXOPT='-O0' trace_mem=1 # Trace all mem accesses with '-trace'"; \
	echo " "
	@echo "Example builds with test runs:"; \
	echo " $(MAKE) -j all   # Normal full API and stencil tests"; \
	echo " $(MAKE) -j all YK_CXX=g++ mpi=0 ranks=1   # Test g++ w/o MPI"; \
	echo " $(MAKE) -j all YK_CXXOPT=-O1 ranks=4   # Test on 4 ranks"
