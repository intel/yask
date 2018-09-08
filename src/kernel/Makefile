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

# Makefile for the YASK stencil kernel.
# Type 'make help' for usage.

# See top-level YASK Makefile for explanation of vars commonly set via the
#   command-line that control kernel configuration and performance.

# Initial defaults.
stencil			:=	iso3dfd
arch			:=	snb
mpi			:=	1
numa			:=	1
real_bytes		:=	4
radius			:=	2
allow_new_grid_types	:=	1

# Defaults based on stencil type (and arch for some stencils).
ifeq ($(stencil),)
 $(error Stencil not specified)

else ifeq ($(stencil),3axis)
 MACROS		+=	MAX_EXCH_DIST=1
 radius		:=	6

else ifeq ($(stencil),9axis)
 MACROS		+=	MAX_EXCH_DIST=2
 radius		:=	4

else ifeq ($(stencil),3plane)
 MACROS		+=	MAX_EXCH_DIST=2
 radius		:=	3

else ifeq ($(stencil),cube)
 MACROS		+=	MAX_EXCH_DIST=3
 radius		:=	2

else ifneq ($(findstring iso3dfd,$(stencil)),)
 MACROS				+=	MAX_EXCH_DIST=1
 radius				:=	8
 def_rank_args			:=	-d 1024 -dx 512 # assume 2 ranks/node in 'x'.
 def_pad_args			:=	-ep 1
 ifeq ($(arch),knl)
  def_rank_args			:=	-d 1024 # assume 1 rank/node.
  fold_4byte			:=	x=2,y=8
  cluster			:=	x=2
  def_block_args		:=	-bx 160 -by 256 -bz 96
  pfd_l1			:=	1
  pfd_l2			:=	0
 else ifneq ($(filter $(arch),hsw bdw),)
  fold_4byte			:=	x=8
  def_thread_divisor		:=	2
  def_block_threads		:=	1
  def_block_args		:=	-bx 296 -by 5 -bz 290
  cluster			:=	z=2
  pfd_l1			:=	0
  pfd_l2			:=	0
 else ifneq ($(filter $(arch),skx skl),)
  fold_4byte			:=	x=4,y=4
  def_thread_divisor		:=	1
  def_block_threads		:=	2
  def_block_args		:=	-bx 108	 -by 28 -bz 132
  cluster			:=	x=1
  pfd_l1			:=	0
  pfd_l2			:=	2
 endif

else ifneq ($(findstring awp,$(stencil)),)
 ifeq ($(stencil),awp)
  time_alloc			:=	1 # only allowed w/o surface conditions.
 endif
 def_block_args			:=	-b 32
 YC_FLAGS			+=	-min-es 1
 def_rank_args			:=	-dx 512 -dy 1024 -dz 128 # assume 2 ranks/node in 'x'.
 def_pad_args			:=	-ep 1
 ifeq ($(arch),knl)
  def_rank_args			:=	-dx 1024 -dy 1024 -dz 128 # assume 1 rank/node.
  fold_4byte			:=	x=4,y=4
  def_thread_divisor		:=	2
  def_block_threads		:=	4
  def_block_args		:=	-b 48 -bx 112
  pfd_l1			:=	1
  pfd_l2			:=	0
 else ifneq ($(filter $(arch),hsw bdw),)
  omp_block_schedule		:=	dynamic,1
  fold_4byte			:=	x=8
  cluster			:=	y=2
  def_block_args		:=	-bx 128 -by 16 -bz 32 
  more_def_args			+=	-sbx 32 -sby 2 -sbz 32
  pfd_l1			:=	1
  pfd_l2			:=	2
 else ifneq ($(filter $(arch),skx skl),)
  fold_4byte			:=	x=2,y=8
  def_block_args		:=	-b 32 -bx 96
  pfd_l1			:=	1
  pfd_l2			:=	0
 endif

else ifneq ($(findstring ssg,$(stencil)),)
 time_alloc		:=	1
 def_rank_args		:=	-d 512
 ifneq ($(filter $(arch),skx skl),)
  def_rank_args		:=	-d 640 -dx 320
  fold_4byte		:=	x=4,y=4
  def_block_args	:=	-bx 96 -by 16 -bz 80
  def_block_threads	:=	2
 endif

else ifneq ($(findstring fsg,$(stencil)),)
 time_alloc		:=	1
 def_rank_args		:=	-d 256
 ifeq ($(arch),knl)
  omp_region_schedule	:=	guided
  def_block_args	:=	-b 16
  def_thread_divisor	:=	4
  def_block_threads	:=	2
  pfd_l1		:=	0
  pfd_l2		:=	2
 else ifneq ($(filter $(arch),skx skl),)
  def_rank_args		:=	-d 384 -dx 192
  fold_4byte		:=	x=4,y=4
  def_block_args	:=	-bx 188 -by 12 -bz 24
  def_block_threads	:=	1
 endif

else ifeq ($(stencil),tti)
 MACROS			+=	MAX_EXCH_DIST=3
 radius			:=	2
 def_rank_args		:=	-d 512
 ifneq ($(filter $(arch),skx skl),)
  def_rank_args		:=	-d 640 -dx 320
  fold_4byte		:=	x=4,y=4
  def_block_args	:=	-bx 80 -by 16 -bz 40
  def_block_threads	:=	1
 endif

else ifeq ($(stencil),stream)
 MACROS		+=	MAX_EXCH_DIST=0
 radius		:=	2
 cluster	:=	x=2

endif # stencil-specific.

# Defaut settings based on architecture.
# (Use '?=' to avoid replacing above settings.)
ifeq ($(arch),knc)

 ISA		?=	-mmic
 MACROS		+=	USE_INTRIN512
 YC_TARGET	?=	knc
 def_block_threads  ?=	4
 pfd_l1		?=	1
 pfd_l2		?=	2

else ifeq ($(arch),knl)

 ISA		?=	-xMIC-AVX512
 GCXX_ISA	?=	-march=knl
 MACROS		+=	USE_INTRIN512 USE_RCP28 NUMA_PREF=1
 YC_TARGET	?=	avx512
 def_block_args ?=	-b 96
 def_block_threads ?=	8
 pfd_l1		?=	1
 pfd_l2		?=	0

else ifneq ($(filter $(arch),skx skl),)

 ISA		?=	-xCORE-AVX512
 GCXX_ISA	?=	-march=knl -mno-avx512er -mno-avx512pf
 MACROS		+=	USE_INTRIN512
 YC_TARGET	?=	avx512

else ifneq ($(filter $(arch),hsw bdw),)

 ISA		?=	-xCORE-AVX2
 GCXX_ISA	?=	-march=haswell
 MACROS		+=	USE_INTRIN256
 YC_TARGET	?=	avx2

else ifeq ($(arch),ivb)

 ISA		?=	-xCORE-AVX-I
 GCXX_ISA	?=	-march=ivybridge
 MACROS		+=	USE_INTRIN256
 YC_TARGET	?=	avx

else ifeq ($(arch),snb)

 ISA		?=	-xAVX
 GCXX_ISA	?=	-march=sandybridge
 MACROS		+=	USE_INTRIN256
 YC_TARGET	?=	avx

else ifeq ($(arch),intel64)

 ISA		?=	-xHOST
 GCXX_ISA	?=	-march=native
 YC_TARGET	?=	cpp

else

$(error Architecture not recognized; use arch=knl, knc, skl, hsw, bdw, ivb, snb, or intel64 (no explicit vectorization))

endif # arch-specific.

# general defaults for vars if not set above.
streaming_stores	?=	0
omp_par_for		?=	omp parallel for
omp_region_schedule	?=	dynamic,1
omp_block_schedule	?=	static,1
omp_misc_schedule	?=	guided
def_thread_divisor	?=	1
def_block_threads	?=	2
real_bytes		?=	4
def_rank_args		?=	-d 128
def_block_args		?=	-b 64
cluster			?=	x=1
pfd_l1			?=	0
pfd_l2			?=	2

# default folding depends on HW vector size.
ifneq ($(findstring INTRIN512,$(MACROS)),)  # 512 bits.

 # 16 SP floats.
 fold_4byte	?=	x=4,y=4

 # 8 DP floats.
 fold_8byte	?=	x=4,y=2

else ifneq ($(findstring INTRIN256,$(MACROS)),)	 # 256 bits.

 # 8 SP floats.
 fold_4byte	?=	x=8

 # 4 DP floats.
 fold_8byte	?=	x=4

else

 fold_4byte	?=	x=1
 fold_8byte	?=	x=1

endif # not 512 bits.

# Select fold based on size of reals.
fold	:=	$(fold_$(real_bytes)byte) # e.g., fold_4byte

######## End of vars that control the function and performance of the kernel.

# Common settings.
YASK_BASE	?=	$(abspath ../..)
include $(YASK_BASE)/src/common/common.mk

# The remainder of this file specifies how to build and test the kernel.
# Vars starting with 'YC_' apply to the stencil compiler.
# Vars starting with 'YK_' apply to the stencil kernel.

# Flags passed to stencil compiler.
YC_FLAGS	+=	-stencil $(stencil) -elem-bytes $(real_bytes) -cluster $(cluster) -fold $(fold)
ifneq ($(eqs),)
 YC_FLAGS	+=	-eq-bundles $(eqs)
endif
ifneq ($(radius),)
 YC_FLAGS	+=	-radius $(radius)
endif
ifneq ($(halo),)
 YC_FLAGS	+=	-halo $(halo)
endif
ifneq ($(time_alloc),)
 YC_FLAGS	+=	-step-alloc $(time_alloc)
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
YK_ARCH_DIR	:=	$(YK_OUT_DIR)/$(arch)
YK_OBJ_DIR	:=	$(YK_ARCH_DIR)/obj

# Dirs for stencil-specific output files.
YK_TAG		:=	$(stencil).$(arch)
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
YK_EXT_SRC_NAMES :=	factory grid_apis context stencil_calc setup \
			realv_grids new_grid settings generic_grids auto_tuner
YK_OBJS		:=	$(addprefix $(YK_OBJ_DIR)/,$(addsuffix .o,$(YK_SRC_NAMES) $(COMM_SRC_NAMES)))
YK_EXT_OBJS	:=	$(addprefix $(YK_EXT_OBJ_DIR)/,$(addsuffix .o,$(YK_EXT_SRC_NAMES)))
YK_CODE_FILE	:=	$(YK_GEN_DIR)/yask_stencil_code.hpp
YK_GEN_HEADERS	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_rank_loops.hpp \
				yask_region_loops.hpp \
				yask_block_loops.hpp \
				yask_sub_block_loops.hpp \
				yask_misc_loops.hpp \
				yask_layout_macros.hpp \
				yask_layouts.hpp \
				yask_grid_code.hpp) \
				$(YK_CODE_FILE)
YK_INC_DIRS	:=	$(INC_DIR) $(YK_LIB_SRC_DIR) $(COMM_DIR)
YK_INC_GLOB	:=	$(addsuffix /*.hpp,$(YK_INC_DIRS))
YK_API_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_test.exe
YK_GRID_TEST_EXEC :=	$(BIN_OUT_DIR)/$(YK_BASE)_grid_test.exe
YK_API_TEST_EXEC_WITH_EXCEPTION :=	$(BIN_OUT_DIR)/$(YK_BASE)_api_exception_test.exe
MAKE_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).make-report.txt
YC_REPORT_FILE :=	$(BUILD_OUT_DIR)/$(YK_EXT_BASE).yask_compiler-report.txt

# File-related macros.
MACROS		+= 	YK_CODE_FILE=$(YK_CODE_FILE)

# YASK loop compiler.
GEN_LOOPS	:=	$(UTILS_BIN_DIR)/gen_loops.pl

# Other scripts and cmds.
GEN_LAYOUTS	:=	$(UTILS_BIN_DIR)/gen_layouts.pl
GET_LOOP_STATS	:=	$(UTILS_BIN_DIR)/get_loop_stats.pl

# Compiler and default flags.
ifeq ($(mpi),1)
 YK_CXX		:=	mpiicc
else
 YK_CXX		:=	icc
endif
YK_CXXFLAGS	:=	-g -std=c++11 -Wall
YK_CXXFLAGS	+=	$(addprefix -I,$(YK_INC_DIRS) $(YK_GEN_DIR))
OMPFLAGS	+=	-fopenmp
YK_CXXOPT	?=	-O3

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

# Turn off prefetching at O0 or O1.
ifneq ($(filter -O0 -O1,$(YK_CXXOPT)),)
 pfd_l1		:=	0
 pfd_l2		:=	0
endif

# Turn on checking at O0.
ifneq ($(filter -O0,$(YK_CXXOPT)),)
 MACROS		+=	CHECK
endif

# Set MACROS based on individual makefile vars.
MACROS		+=	PFD_L1=$(pfd_l1) PFD_L2=$(pfd_l2)
ifeq ($(streaming_stores),1)
 MACROS		+=	USE_STREAMING_STORE
endif
MACROS		+=	ALLOW_NEW_GRIDS=$(allow_new_grid_types)

# Default cmd-line args.
DEF_ARGS	+=	-thread_divisor $(def_thread_divisor)
DEF_ARGS	+=	-block_threads $(def_block_threads)
DEF_ARGS	+=	$(def_rank_args) $(def_block_args) $(def_pad_args) $(more_def_args) 
YK_CXXFLAGS	+=	-DDEF_ARGS='"$(DEF_ARGS) $(EXTRA_DEF_ARGS)"'

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
MACROS		+=	ARCH_$(ARCH)

# MPI settings.
ifeq ($(mpi),1)
 MACROS		+=	USE_MPI
endif

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
ifneq ($(VTUNE_AMPLIFIER_2018_DIR),)
 VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_2018_DIR)
else ifneq ($(VTUNE_AMPLIFIER_XE_2017_DIR),)
 VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_XE_2017_DIR)
else
 VTUNE_DIR	:=	$(VTUNE_AMPLIFIER_XE_2016_DIR)
endif
YK_CXXFLAGS	+=	-I$(VTUNE_DIR)/include
YK_LIBS		+=	$(VTUNE_DIR)/lib64/libittnotify.a
endif

# compiler-specific settings
ifneq ($(filter %icpc %icc,$(notdir $(YK_CXX))),)  # Intel compiler

 YK_CXXFLAGS	+=	$(ISA) -debug extended -restrict -ansi-alias -fno-alias
 YK_CXXFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt \
			-no-prec-div -fp-model fast=2 -fno-protect-parens -ftz \
			-fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 YK_CXXFLAGS	+=	-no-diag-message-catalog
 YK_CXXFLAGS	+=	-vec-threshold0
 #YK_CXXFLAGS	+=	-qoverride-limits
 #YK_CXXFLAGS	+=	-Fa
 #YK_CXXFLAGS	+=	-qopt-report=5
 #YK_CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 YK_CXX_VER_CMD :=	$(YK_CXX) -V

# Use 512-bit code on SKL.
# (Commented out because it doesn't help performance, and it breaks Devito.)
ifneq ($(filter $(arch),skx skl),)
 #YK_CXXFLAGS	+=	-qopt-zmm-usage=high
endif

else # not Intel compiler
 YK_CXXFLAGS	+=	$(GCXX_ISA) -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable

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

# Number of stencil/grid dims extracted from YASK compiler output.
NSDIMS		:=	`awk '/NUM_STENCIL_DIMS/ {print $$NF}' $(YK_CODE_FILE)`
NGDIMS		:=	`awk '/NUM_GRID_DIMS/ {print $$NF}' $(YK_CODE_FILE)`

######## Loop-compiler configuration:
# The loop indices range from 0..N-1.
# Dim 0 is the step dim, usually time.
# The step loop is handled outside of the generated loops,
# so the following loop codes do not scan over dim 0.
# Dims 1..N-1 are the domain dims, usually spatial.
# Thus, N-1 is the inner-most dim.
# For best perf, this should be the unit-stride dim in the grids.

# Rank loops break up the whole rank into smaller regions.  In order for
# temporal wavefronts to operate properly, the order of spatial dimensions
# may be changed, but the scanning paths must have strictly incrementing
# indices. Those that do not (e.g., grouped, serpentine, square-wave) may
# *not* be used here when using temporal wavefronts. The time loop may be
# found in StencilEquations::run_solution().
RANK_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar rank_idxs
RANK_LOOP_ORDER		?=	1 .. N-1
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop($(RANK_LOOP_ORDER)) \
				{ $(RANK_LOOP_INNER_MODS) call(calc_region(bp)); }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.	 The region time loops are not coded here to
# allow for proper spatial skewing for temporal wavefronts. The time loop
# may be found in StencilEquations::calc_region().
REGION_LOOP_OPTS	?=	-ndims $(NSDIMS) -inVar region_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_region_schedule)) proc_bind(spread)'
REGION_LOOP_OUTER_MODS	?=	grouped omp
REGION_LOOP_ORDER	?=	1 .. N-1
REGION_LOOP_CODE	?=	$(REGION_LOOP_OUTER_MODS) loop($(REGION_LOOP_ORDER)) { \
				$(REGION_LOOP_INNER_MODS) call(calc_block(bp, phase, shift_num)); }

# Block loops break up a block into sub-blocks.	 The 'omp' modifier creates
# a *nested* OpenMP loop so that each sub-block is assigned to a nested OpenMP
# thread.  There is no time loop because threaded temporal blocking is
# not yet supported.
BLOCK_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar block_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)' \
				-callPrefix 'sg->'
BLOCK_LOOP_OUTER_MODS	?=	grouped omp
BLOCK_LOOP_ORDER	?=	1 .. N-1
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_OUTER_MODS) loop($(BLOCK_LOOP_ORDER)) { \
				$(BLOCK_LOOP_INNER_MODS) call(calc_sub_block(thread_idx)); }

# Sub-block loops break up a sub-block into clusters or vectors.  These loops
# are run by a single OMP thread.  The N-1 (inner) loop is generated by the
# stencil compiler.  There is no time loop because threaded temporal
# blocking is not yet supported.  The indexes in this loop are 'normalized',
# i.e., vector units and rank-relative.
SUB_BLOCK_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar norm_sub_block_idxs
SUB_BLOCK_LOOP_OUTER_MODS	?=
SUB_BLOCK_LOOP_ORDER		?=	1 .. N-2
SUB_BLOCK_LOOP_CODE		?=	$(SUB_BLOCK_LOOP_OUTER_MODS) loop($(SUB_BLOCK_LOOP_ORDER)) { \
					$(SUB_BLOCK_LOOP_INNER_MODS) call(calc_inner_loop(thread_idx)); }

# General-purpose parallel loop.
# Nested OpenMP is not used here because there is no sharing between threads.
# TODO: Consider using nested OpenMP to hide more latency.
MISC_LOOP_OPTS		?=	-ndims $(NSDIMS) -inVar misc_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_misc_schedule)) proc_bind(spread)'
MISC_LOOP_OUTER_MODS	?=	omp
MISC_LOOP_ORDER		?=	1 .. N-1
MISC_LOOP_CODE		?=	$(MISC_LOOP_OUTER_MODS) loop($(MISC_LOOP_ORDER)) \
				$(MISC_LOOP_INNER_MODS) { call(misc_fn); }

# Compile rules.

$(YK_OBJ_DIR)/%.o: $(COMM_DIR)/%.cpp $(YK_INC_GLOB)
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
	echo "*** Run command:" $(YK_SCRIPT) "-stencil" $(stencil) "-arch" $(arch) "[options]"

$(YK_LIB): $(YK_OBJS) $(YK_EXT_OBJS)
	- rm -f $(MAKE_REPORT_FILE)
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^ $(YK_LIBS)
	@ls -l $@

$(YK_EXEC): yask_main.cpp $(YK_LIB)
	- rm -f $(MAKE_REPORT_FILE)
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

$(YK_GEN_DIR)/yask_sub_block_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -output $@ $(SUB_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_SUB_BLOCK_LOOP_OPTS) "$(SUB_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_misc_loops.hpp: $(GEN_LOOPS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$< -output $@ $(MISC_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_MISC_LOOP_OPTS) "$(MISC_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -m $(NGDIMS) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_layouts.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -d $(NGDIMS) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_grid_code.hpp: $(GEN_LAYOUTS) $(YK_CODE_FILE)
	$(MKDIR) $(YK_GEN_DIR)
	$(PERL) $< -g $(NGDIMS) > $@

$(YK_CODE_FILE): $(YC_EXEC)
	$(MKDIR) $(dir $@)
	$(RUN_PREFIX) $< $(YC_FLAGS) $(EXTRA_YC_FLAGS) -p $(YC_TARGET) $@ | tee $(YC_REPORT_FILE) 2>&1
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
	  -I$(INC_DIR) -I$(COMM_DIR) -I$(COMM_DIR)/swig \
	  -c++ -python -o $@ -outdir $(PY_OUT_DIR) -builtin $<

$(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o: $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.cpp
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<
	@ls -l $@

$(YK_PY_LIB): $(YK_OBJS)  $(YK_EXT_OBJS) $(YK_SWIG_OUT_DIR)/yask_kernel_api_wrap.o
	$(MKDIR) $(dir $@)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^ $(YK_LIBS)
	@ls -l $@

# Simple tests.

$(YK_GRID_TEST_EXEC): $(YK_TEST_SRC_DIR)/grid_test.cpp $(YK_LIB)
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
cxx-yk-grid-test: $(YK_GRID_TEST_EXEC)
	@echo '*** Running the C++ YASK grid test...'
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
val1	:=	-d 48 -dt 2 -b 16 -bt 1
val2	:=	-d 63 -dt 2 -b 24 -bt 1 -r 32 -rt 2
val3	:=	-d 61 -dt 2 -b 24 -bt 2
val4	:=	-d 64 -dt 2 -b 32 -bt 2 -r 48 -rt 2
ranks	:=	2

# Run the kernel binary using several combos of sizes and ranks.
yk-tests:
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -v $(val1)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -v $(val2)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -v $(val3)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -v $(val4)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -ranks $(ranks) -v $(val1)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -ranks $(ranks) -v $(val2)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -ranks $(ranks) -v $(val3)
	$(YK_SCRIPT) -stencil $(stencil) -arch $(arch) -ranks $(ranks) -v $(val4)

# Run the default YASK compiler and kernel.
yc-and-yk-test: $(YK_EXEC) $(YK_SCRIPT)
	$(MAKE) yk-tests

# Run the YASK kernel test without implicity using the YASK compiler.
yk-test-no-yc: kernel-no-yc $(YK_SCRIPT)
	$(MAKE) yk-tests

# Run the kernel API tests for C++ and Python with and w/o expected exceptions.
api-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-api-test real_bytes=8 stencil=iso3dfd
	$(MAKE) clean; $(MAKE) py-yk-api-test stencil=iso3dfd
	$(MAKE) clean; $(MAKE) cxx-yk-api-test-with-exception real_bytes=8 stencil=iso3dfd
	$(MAKE) clean; $(MAKE) py-yk-api-test-with-exception stencil=iso3dfd

# Run several stencils using built-in validation.
# NB: set arch var as applicable.
# NB: save some time by using YK_CXXOPT=-O2.
# These tests are focused on the kernel and not the compiler.
# For testing both the kernel and compiler in various combinations,
# run the tests from the top-level Makefile.
stencil-tests:
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_1d fold=x=4
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_2d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_3d fold=x=2,z=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_4d fold=w=2,x=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_step_cond_1d fold=x=4
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_subdomain_1d fold=x=4
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_subdomain_3d fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_scratch1 fold=x=4
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=test_scratch2 fold=x=2,z=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=9axis fold=x=2,z=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=3axis fold=x=2,y=2 cluster=x=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=3axis fold=x=2,y=2 cluster=z=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=3plane fold=y=2,z=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=cube fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=iso3dfd fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=ssg fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=awp_elastic fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=fsg_abc fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=fsg2 fold=x=2,y=2
	$(MAKE) clean; $(MAKE) yc-and-yk-test real_bytes=8 stencil=tti fold=x=2,y=2

all-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-grid-test stencil=test_3d fold=x=4,y=2
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
	- find . $(COMM_DIR) -name '*.*.o' -print -delete
	- find . $(COMM_DIR) -name '*.optrpt' -print -delete

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
	$(YK_CXX) --version; \
	echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""; \
	echo BLOCK_LOOP_INNER_MODS="\"$(BLOCK_LOOP_INNER_MODS)\""; \
	echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""; \
	echo BLOCK_LOOP_ORDER="\"$(BLOCK_LOOP_ORDER)\""; \
	echo BLOCK_LOOP_OUTER_MODS="\"$(BLOCK_LOOP_OUTER_MODS)\""; \
	echo BLOCK_LOOP_OUTER_VARS="\"$(BLOCK_LOOP_OUTER_VARS)\""; \
	echo EXTRA_DEF_ARGS=$(EXTRA_DEF_ARGS); \
	echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""; \
	echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""; \
	echo EXTRA_YK_CXXFLAGS="\"$(EXTRA_YK_CXXFLAGS)\""; \
	echo ISA=$(ISA); \
	echo MACROS="\"$(MACROS)\""; \
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
	echo YC_TARGET="\"$(YC_TARGET)\""; \
	echo YK_BASE=$(YK_BASE); \
	echo YK_CXX=$(YK_CXX); \
	echo YK_CXXFLAGS="\"$(YK_CXXFLAGS)\""; \
	echo YK_CXXOPT=$(YK_CXXOPT); \
	echo YK_EXT_TAG=$(YK_EXT_TAG); \
	echo arch=$(arch); \
	echo cluster=$(cluster); \
	echo def_block_args=$(def_block_args); \
	echo def_block_threads=$(def_block_threads); \
	echo def_pad_args=$(def_pad_args); \
	echo def_rank_args=$(def_rank_args); \
	echo def_thread_divisor=$(def_thread_divisor); \
	echo fold=$(fold); \
	echo host=`hostname`; \
	echo more_def_args=$(more_def_args); \
	echo omp_block_schedule=$(omp_block_schedule); \
	echo omp_misc_schedule=$(omp_misc_schedule); \
	echo omp_region_schedule=$(omp_region_schedule); \
	echo pfd_l1=$(pfd_l1); \
	echo pfd_l2=$(pfd_l2); \
	echo radius=$(radius); \
	echo real_bytes=$(real_bytes); \
	echo stencil=$(stencil); \
	echo streaming_stores=$(streaming_stores)

# Print stats on inner SIMD loops from asm file.
code-stats:
	@echo "Code stats for stencil computation:"
	$(PERL) $(GET_LOOP_STATS) -t=$(YK_CODE_FILE) factory.s

# Print some usage info.
help:
	@echo "Example performance builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=iso3dfd"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=awp EXTRA_YK_CXXFLAGS=-qopt-report"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=skl stencil=3axis fold='x=1,y=2,z=4' cluster='x=2'"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=hsw stencil=3axis radius=4 real_bytes=8 # DP FP"; \
	echo " "; \
	echo "Example performance builds of kernel API for C++ and Python apps:"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=iso3dfd yk-api"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=skl stencil=awp yk-api"; \
	echo " "; \
	echo "Example debug builds of kernel cmd-line tool:"; \
	echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' EXTRA_MACROS='CHECK' # No OpenMP or MPI, internal checking"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=test_2d mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' EXTRA_MACROS='CHECK TRACE'	 # TRACE is a useful debug setting!"; \
	echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' mpi=0 YK_CXX=g++ OMPFLAGS='' YK_CXXOPT='-O0' EXTRA_MACROS='CHECK TRACE TRACE_MEM TRACE_INTRINSICS'"; \
	echo " "; \
	echo "Example builds with test runs:"; \
	echo " $(MAKE) -j all # Normal full API and stencil tests"; \
	echo " $(MAKE) -j all YK_CXXOPT=-O2 YK_CXX=g++ mpi=0 ranks=1 # g++ w/o MPI"; \
	echo " $(MAKE) -j all YK_CXXOPT=-O1 ranks=3 EXTRA_MACROS='CHECK' # Run 3 ranks w/checking"
