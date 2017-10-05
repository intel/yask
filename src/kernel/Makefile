##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2017, Intel Corporation
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
stencil		=	iso3dfd
arch		=	snb
mpi		=	1
real_bytes	=	4
radius		=	1

# Defaults based on stencil type (and arch for some stencils).
ifeq ($(stencil),)
 $(error Stencil not specified)

else ifeq ($(stencil),3axis)
 MACROS		+=	MAX_EXCH_DIST=1
 radius		=	6

else ifeq ($(stencil),9axis)
 MACROS		+=	MAX_EXCH_DIST=2
 radius		=	4

else ifeq ($(stencil),3plane)
 MACROS		+=	MAX_EXCH_DIST=2
 radius		=	3

else ifeq ($(stencil),cube)
 MACROS		+=	MAX_EXCH_DIST=3
 radius		=	2

else ifeq ($(stencil),iso3dfd)
 MACROS				+=	MAX_EXCH_DIST=1
 radius				=	8
 def_rank_args			=	-d 1024 -dx 512 # assume 2 ranks/node in 'x'.
 def_pad_args			=	-ep 1
 ifeq ($(arch),knl)
  def_rank_args			=	-d 1024 # assume 1 rank/node.
  fold_4byte			=	x=2,y=8
  cluster			=	x=2
  def_block_args		=	-b 96 -bx 192
 else ifeq ($(arch),hsw)
  fold_4byte			=	x=8
  def_thread_divisor		=	2
  def_block_threads		=	1
  def_block_args		=	-bx 296 -by 5 -bz 290
  cluster			=	z=2
  prefetch			=
 else ifeq ($(arch),skx)
  fold_4byte			=	x=4,y=4
  def_thread_divisor		=	1
  def_block_threads		=	2
  def_block_args		=	-b 64
  cluster			=	z=2
  prefetch			=	prefetch(L1)
 endif

else ifneq ($(findstring awp,$(stencil)),)
 time_alloc			=	1
 eqs				=	velocity=vel,stress=str
 def_block_args			=	-b 32
 YC_FLAGS			+=	-min-es 1
 def_rank_args			=	-dx 512 -dy 1024 -dz 128 # assume 2 ranks/node in 'x'.
 def_pad_args			=	-ep 1
 ifeq ($(arch),knl)
  def_rank_args			=	-dx 1024 -dy 1024 -dz 128 # assume 1 rank/node.
  fold_4byte			=	x=4,y=4
  def_thread_divisor		=	2
  def_block_threads		=	4
  def_block_args		=	-b 48 -bx 112
 else ifeq ($(arch),hsw)
  prefetch			=	prefetch(L1,L2)
  omp_block_schedule		=	dynamic,1
  fold_4byte			=	x=8
  cluster			=	y=2
  def_pad_args			=	-epx 1 -epy 1 -epz 0
  def_block_args		=	-bx 128 -by 16 -bz 32 
  more_def_args			+=	-sbx 32 -sby 2 -sbz 32
 else ifeq ($(arch),skx)
  fold_4byte			=	x=2,y=8
  def_block_args		=	-b 32 -bx 96
  prefetch			=	prefetch(L1)
 endif

else ifneq ($(findstring ssg,$(stencil)),)
 time_alloc	=	1
 eqs		=	v_bl=v_bl,v_tr=v_tr,v_tl=v_tl,s_br=s_br,s_bl=s_bl,s_tr=s_tr,s_tl=s_tl

else ifneq ($(findstring fsg,$(stencil)),)
 time_alloc	=	1
 eqs		=      v_br=v_br,v_bl=v_bl,v_tr=v_tr,v_tl=v_tl,s_br=s_br,s_bl=s_bl,s_tr=s_tr,s_tl=s_tl
 ifeq ($(arch),knl)
  omp_region_schedule  	=	guided
  def_block_args  	=	-b 16
  def_thread_divisor	=	4
  def_block_threads	=	1
  prefetch		=	prefetch(L2)
 endif

else ifeq ($(stencil),stream)
 MACROS		+=	MAX_EXCH_DIST=0
 radius		=	2
 cluster	=	x=2

endif # stencil-specific.

# Defaut settings based on architecture.
# (Use '?=' to avoid replacing above settings.)
ifeq ($(arch),knc)

 ISA		?= 	-mmic
 MACROS		+=	USE_INTRIN512
 YC_TARGET  	?=	knc
 def_block_threads  ?=	4
 prefetch	?=	prefetch(L1,L2)

else ifeq ($(arch),knl)

 ISA		?=	-xMIC-AVX512
 GCXX_ISA	?=	-march=knl
 MACROS		+=	USE_INTRIN512 USE_RCP28
 YC_TARGET  	?=	avx512
 def_block_args	?=	-b 96
 def_block_threads ?=	8
 prefetch	?=	prefetch(L1)

else ifeq ($(arch),skx)

 ISA		?=	-xCORE-AVX512
 GCXX_ISA	?=	-march=knl -mno-avx512er -mno-avx512pf
 MACROS		+=	USE_INTRIN512
 YC_TARGET  	?=	avx512

else ifeq ($(arch),hsw)

 ISA		?=	-xCORE-AVX2
 GCXX_ISA	?=	-march=haswell
 MACROS		+=	USE_INTRIN256
 YC_TARGET  	?=	avx2

else ifeq ($(arch),ivb)

 ISA		?=	-xCORE-AVX-I
 GCXX_ISA	?=	-march=ivybridge
 MACROS		+=	USE_INTRIN256
 YC_TARGET  	?=	avx

else ifeq ($(arch),snb)

 ISA		?=	-xAVX
 GCXX_ISA	?=	-march=sandybridge
 MACROS		+= 	USE_INTRIN256
 YC_TARGET  	?=	avx

else ifeq ($(arch),intel64)

 ISA		?=	-xHOST
 GCXX_ISA       ?=      -march=native
 YC_TARGET	?=	cpp

else

$(error Architecture not recognized; use arch=knl, knc, skx, hsw, ivb, snb, or intel64 (no explicit vectorization))

endif # arch-specific.

# general defaults for vars if not set above.
streaming_stores	?= 	0
omp_par_for		?=	omp parallel for
omp_region_schedule	?=	dynamic,1
omp_block_schedule	?=	static,1
omp_gen_schedule	?=	guided
prefetch		?=	prefetch(L2)
def_thread_divisor	?=	1
def_block_threads	?=	2
real_bytes		?=	4
def_rank_args		?=	-d 128
def_block_args		?=	-b 64
def_pad_args		?=	-ep 1
cluster			?=	x=1
pfd_l1			?=	1
pfd_l2			?=	2
max_dims		?=	5  # max grid dims >= max stencil dims.

# default folding depends on HW vector size.
ifneq ($(findstring INTRIN512,$(MACROS)),)  # 512 bits.

 # 16 SP floats.
 fold_4byte	?=	x=4,y=4

 # 8 DP floats.
 fold_8byte	?=	x=4,y=2

else  # not 512 bits (assume 256).

 # 8 SP floats.
 fold_4byte	?=	x=8

 # 4 DP floats.
 fold_8byte	?=	x=4

endif # not 512 bits.

# Select fold based on size of reals.
fold	= 	$(fold_$(real_bytes)byte) # e.g., fold_4byte

######## Loop-compiler configuration:
# The loop indices range from 0..N-1.
# Dim 0 is the step dim, usually time.
# The step loop is handled outside of the generated loops,
# so the following loop codes do not scan over dim 0.
# Dims 1..N-1 are the domain dims, usually spatial.
# Thus, N-1 is the inner-most dim.
# For best perf, this should be the unit-stride dim in the grids.

# File with number of dims extracted from YASK compiler output.
YK_DIMS_FILE		:=	num_dims.$(stencil).txt
NDIMS_OPT		:=	`cat $(YK_DIMS_FILE)`

# Rank loops break up the whole rank into smaller regions.  In order for
# temporal wavefronts to operate properly, the order of spatial dimensions
# may be changed, but the scanning paths must have strictly incrementing
# indices. Those that do not (e.g., grouped, serpentine, square-wave) may
# *not* be used here when using temporal wavefronts. The time loop may be
# found in StencilEquations::run_solution().
RANK_LOOP_OPTS		?=	$(NDIMS_OPT) -inVar rank_idxs
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop(1 .. N-1) \
				{ $(RANK_LOOP_INNER_MODS) calc(region(eqGroup_ptr)); }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.  The region time loops are not coded here to
# allow for proper spatial skewing for temporal wavefronts. The time loop
# may be found in StencilEquations::calc_region().
REGION_LOOP_OPTS	?=     	$(NDIMS_OPT) -inVar region_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_region_schedule)) proc_bind(spread)' \
				-calcPrefix 'eg->calc_'
REGION_LOOP_OUTER_MODS	?=	grouped omp
REGION_LOOP_CODE	?=	$(REGION_LOOP_OUTER_MODS) loop(1 .. N-1) { \
				$(REGION_LOOP_INNER_MODS) calc(block); }

# Block loops break up a block into sub-blocks.  The 'omp' modifier creates
# a *nested* OpenMP loop so that each sub-block is assigned to a nested OpenMP
# thread.  There is no time loop because threaded temporal blocking is
# not yet supported.
BLOCK_LOOP_OPTS		?=     	$(NDIMS_OPT) -inVar block_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)'
BLOCK_LOOP_OUTER_MODS	?=	grouped omp
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_OUTER_MODS) loop(1 .. N-1) { \
				$(BLOCK_LOOP_INNER_MODS) calc(sub_block); }

# Sub-block loops break up a sub-block into vector clusters.  These loops
# are run by a single OMP thread.  The N-1 (inner) loop is generated by the
# stencil compiler.  There is no time loop because threaded temporal
# blocking is not yet supported.  The indexes in this loop are 'normalized',
# i.e., vector units and rank-relative.
SUB_BLOCK_LOOP_OPTS		?=     	$(NDIMS_OPT) -inVar norm_sub_block_idxs
SUB_BLOCK_LOOP_OUTER_MODS	?=	square_wave serpentine
SUB_BLOCK_LOOP_CODE		?=	$(SUB_BLOCK_LOOP_OUTER_MODS) loop(1 .. N-2) { \
					$(SUB_BLOCK_LOOP_INNER_MODS) calc(loop_of_clusters); }

# General-purpose parallel loop.
# Nested OpenMP is not used here because there is no sharing between threads.
# TODO: Consider using nested OpenMP to hide more latency.
GEN_LOOP_OPTS		?=     	$(NDIMS_OPT) -inVar gen_idxs \
				-ompConstruct '$(omp_par_for) schedule($(omp_gen_schedule)) proc_bind(spread)'
GEN_LOOP_OUTER_MODS	?=	omp
GEN_LOOP_CODE		?=	$(GEN_LOOP_OUTER_MODS) loop(1 .. N-1) \
				$(GEN_LOOP_INNER_MODS) { calc(gen()); }

######## End of vars that control the function and performance of the kernel.

# The remainder of this file specifies how to build and test the kernel.
# Vars starting with 'YC_' apply to the stencil compiler.
# Vars starting with 'YK_' apply to the stencil kernel.

# Flags passed to stencil compiler.
YC_FLAGS   	+=	-stencil $(stencil) -elem-bytes $(real_bytes) -cluster $(cluster) -fold $(fold)
ifneq ($(eqs),)
 YC_FLAGS   	+=	-eq $(eqs)
endif
ifneq ($(radius),)
 YC_FLAGS   	+=	-radius $(radius)
endif
ifneq ($(halo),)
 YC_FLAGS   	+=	-halo $(halo)
endif
ifneq ($(time_alloc),)
 YC_FLAGS   	+=	-step-alloc $(time_alloc)
endif

# YASK dirs.
YASK_BASE	:=	$(shell cd ../..; pwd)
LIB_DIR		:=	$(YASK_BASE)/lib
INC_DIR		:=	$(YASK_BASE)/include
BIN_DIR		:=	$(YASK_BASE)/bin
YASK_DIR	:=	$(YASK_BASE)/yask

# OS-specific
ifeq ($(shell uname -o),Cygwin)
  SO_SUFFIX	:=	.dll
  RUN_PREFIX	:=	env PATH="${PATH}:$(LIB_DIR)"
  PYTHON		:= python3
else
  SO_SUFFIX	:=	.so
  RUN_PREFIX	:=
  PYTHON		:=	python
  CXX_PREFIX	:=	time
endif

# Output files and dirs.
YK_BASE		:=	yask_kernel
YK_MODULE	:=	$(YK_BASE)
YK_TAG		:=	$(stencil).$(arch)
YK_BASE2	:=	$(YK_BASE).$(YK_TAG)
YK_EXEC		:=	$(BIN_DIR)/$(YK_BASE2).exe
YK_LIB		:=	$(LIB_DIR)/lib$(YK_BASE2)$(SO_SUFFIX)
YK_PY_LIB	:=	$(YASK_DIR)/_$(YK_MODULE)$(SO_SUFFIX)
YK_PY_MOD	:=	$(YASK_DIR)/$(YK_MODULE).py
YK_API_TEST_EXEC :=	$(BIN_DIR)/$(YK_BASE)_api_test.exe
YK_GRID_TEST_EXEC :=	$(BIN_DIR)/$(YK_BASE)_grid_test.exe
YK_TUPLE_TEST_EXEC :=	$(BIN_DIR)/$(YK_BASE)_tuple_test.exe

MAKE_REPORT_FILE:=	make-report.$(YK_TAG).txt

# Common source.
COMM_DIR	:=	../common
COMM_SRC_NAMES	:=	output tuple
COMM_SRC_BASES	:=	$(addprefix $(COMM_DIR)/,$(COMM_SRC_NAMES))

# Kernel source files and dirs.
YK_SWIG_DIR	:=	./swig
YK_GEN_DIR	:=	./gen
YK_LIB_DIR	:=	./lib
YK_SRC_NAMES	:=	factory new_grid generic_grids realv_grids utils settings context stencil_calc
YK_SRC_BASES	:=	$(addprefix $(YK_LIB_DIR)/,$(YK_SRC_NAMES))
YK_OBJS		:=	$(addsuffix .$(YK_TAG).o,$(YK_SRC_BASES) $(COMM_SRC_BASES))
YK_MACRO_FILE	:=	$(YK_GEN_DIR)/yask_macros.hpp
YK_CODE_FILE	:=	$(YK_GEN_DIR)/yask_stencil_code.hpp
YK_GEN_HEADERS	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_rank_loops.hpp \
				yask_region_loops.hpp \
				yask_block_loops.hpp \
				yask_sub_block_loops.hpp \
				yask_gen_loops.hpp \
				yask_layout_macros.hpp \
				yask_layouts.hpp \
				yask_grid_code.hpp) \
				$(YK_MACRO_FILE) \
				$(YK_CODE_FILE)
YK_INC_DIRS	:=	$(INC_DIR) $(YK_LIB_DIR) ../common
YK_INC_GLOB	:=	$(addsuffix /*.hpp,$(YK_INC_DIRS))
YK_MK_GEN_DIR	:=	mkdir -p -v $(YK_GEN_DIR)

# YASK stencil compiler.
YC_BASE		:=	yask_compiler
YC_EXEC		:=	$(BIN_DIR)/$(YC_BASE).exe
YC_SRC_DIR	:=	../compiler

# YASK loop compiler.
GEN_LOOPS	:=	$(BIN_DIR)/gen_loops.pl

# Other scripts.
GEN_LAYOUTS	:=	$(BIN_DIR)/gen_layouts.pl
GET_LOOP_STATS	:=	$(BIN_DIR)/get_loop_stats.pl

# Compiler and default flags.
ifeq ($(mpi),1)
 YK_CXX		:=	mpiicpc
else
 YK_CXX		:=	icpc
endif
YK_CXXFLAGS	+=   	-g -std=c++11 -Wall
YK_CXXFLAGS	+=	$(addprefix -I,$(YK_INC_DIRS) $(YK_GEN_DIR))
OMPFLAGS	+=	-fopenmp
YK_CXXOPT	?=	-O3

# Add user-defined flags.
YK_CXXFLAGS	+=	$(EXTRA_YK_CXXFLAGS)

# Linker.
YK_LD		:=	$(YK_CXX)
YK_LFLAGS	:=	-lrt -Wl,-rpath=$(LIB_DIR) -L$(LIB_DIR) -l$(YK_BASE2)

# Tools.
SWIG		:=	swig

# Find include path needed for python interface.
# NB: constructing string inside print() to work for python 2 or 3.
PYINC		:= 	$(addprefix -I,$(shell $(PYTHON) -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc() + " " + distutils.sysconfig.get_python_inc(plat_specific=1))'))

RUN_PYTHON	:= 	$(RUN_PREFIX) env PYTHONPATH=$(LIB_DIR) $(PYTHON)

# Set MACROS based on individual makefile vars.
# MACROS and EXTRA_MACROS will be written to a header file.
MACROS		+=	PFDL1=$(pfd_l1) PFDL2=$(pfd_l2)
MACROS		+=	MAX_DIMS=$(max_dims)
ifeq ($(streaming_stores),1)
 MACROS		+=	USE_STREAMING_STORE
endif

# Default cmd-line args.
DEF_ARGS	+=	-thread_divisor $(def_thread_divisor)
DEF_ARGS	+=	-block_threads $(def_block_threads)
DEF_ARGS	+=	$(def_rank_args) $(def_block_args) $(def_pad_args) $(more_def_args) 
MACROS		+=	DEF_ARGS='"$(DEF_ARGS) $(EXTRA_DEF_ARGS)"'

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
MACROS		+= 	ARCH_$(ARCH)

# MPI settings.
ifeq ($(mpi),1)
 MACROS		+=	USE_MPI
endif

# HBW settings.
# TODO: change this to use OS-default location.
ifeq ($(hbw),1)
 MACROS		+=	USE_HBW
 HBW_DIR 	=	$(HOME)/memkind_build
 YK_CXXFLAGS	+=	-I$(HBW_DIR)/include
 YK_LFLAGS	+= 	-lnuma $(HBW_DIR)/lib/libmemkind.a
endif

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
ifneq ($(VTUNE_AMPLIFIER_XE_2018_DIR),)
 VTUNE_DIR	=	$(VTUNE_AMPLIFIER_XE_2018_DIR)
else ifneq ($(VTUNE_AMPLIFIER_XE_2017_DIR),)
 VTUNE_DIR	=	$(VTUNE_AMPLIFIER_XE_2017_DIR)
else
 VTUNE_DIR	=	$(VTUNE_AMPLIFIER_XE_2016_DIR)
endif
YK_CXXFLAGS	+=	-I$(VTUNE_DIR)/include
YK_LFLAGS	+=	$(VTUNE_DIR)/lib64/libittnotify.a
endif

# compiler-specific settings
ifneq ($(findstring icpc,$(notdir $(YK_CXX))),)  # Intel compiler

 YK_CXXFLAGS	+=      $(ISA) -debug extended -Fa -restrict -ansi-alias -fno-alias
 YK_CXXFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt \
			-no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz \
			-fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 #YK_CXXFLAGS	+=	-qoverride-limits
 YK_CXXFLAGS	+=	-vec-threshold0
 #YK_CXXFLAGS	+=      -qopt-report=5
 #YK_CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 YK_CXXFLAGS	+=	-no-diag-message-catalog
 YK_CXX_VER_CMD	:=	$(YK_CXX) -V

 # work around an optimization anomaly.
 MACROS		+=	NO_STORE_INTRINSICS

else # not Intel compiler
 YK_CXXFLAGS	+=	$(GCXX_ISA) -Wno-unknown-pragmas -Wno-unused-variable

endif # compiler.

# Compile with model_cache=1 or 2 to check prefetching.
# Turn off OpenMP when modeling a cache.
ifeq ($(model_cache),1)
 MACROS       	+=      MODEL_CACHE=1
 OMPFLAGS	:=	-qopenmp-stubs
else ifeq ($(model_cache),2)
 MACROS       	+=      MODEL_CACHE=2
 OMPFLAGS	:=	-qopenmp-stubs
endif

# Add in final flags and user-added flags.
YK_CXXFLAGS	+=	$(YK_CXXOPT) $(OMPFLAGS) $(EXTRA_YK_CXXFLAGS)

######## Primary targets & rules
# NB: must set stencil and arch make vars to generate the desired YASK kernel.

default: kernel

# Compile the kernel.
kernel:	$(YK_EXEC) $(MAKE_REPORT_FILE)
	@echo "Binary" $(YK_EXEC) "has been built."
	@echo "See" $(MAKE_REPORT_FILE) "for build information."
	@echo "Run command:" $(BIN_DIR)"/yask.sh -stencil" $(stencil) "-arch" $(arch) "[options]"

%.$(YK_TAG).o: %.cpp $(YK_GEN_HEADERS) $(YK_INC_GLOB)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -fPIC -c -o $@ $<
	@ls -l $@

$(YK_LIB): $(YK_OBJS)
	$(CXX_PREFIX) $(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^
	@ls -l $@

$(YK_EXEC): yask_main.cpp $(YK_LIB)
	$(CXX_PREFIX) $(YK_LD) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@
	@ls -l $@

$(MAKE_REPORT_FILE): $(YK_EXEC)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	$(MAKE) echo-settings >> $@ 2>&1
#$(MAKE) code-stats | tee -a $@

# Generated files.
$(YK_GEN_DIR)/yask_rank_loops.hpp: $(GEN_LOOPS) $(YK_DIMS_FILE)
	$(YK_MK_GEN_DIR)
	$< -output $@ $(RANK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_RANK_LOOP_OPTS) "$(RANK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_region_loops.hpp: $(GEN_LOOPS) $(YK_DIMS_FILE)
	$(YK_MK_GEN_DIR)
	$< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

$(YK_GEN_DIR)/yask_block_loops.hpp: $(GEN_LOOPS) $(YK_DIMS_FILE)
	$(YK_MK_GEN_DIR)
	$< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_sub_block_loops.hpp: $(GEN_LOOPS) $(YK_DIMS_FILE)
	$(YK_MK_GEN_DIR)
	$< -output $@ $(SUB_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_SUB_BLOCK_LOOP_OPTS) "$(SUB_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_gen_loops.hpp: $(GEN_LOOPS) $(YK_DIMS_FILE)
	$(YK_MK_GEN_DIR)
	$< -output $@ $(GEN_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_GEN_LOOP_OPTS) "$(GEN_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: $(GEN_LAYOUTS)
	$(YK_MK_GEN_DIR)
	$< -m $(max_dims) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_layouts.hpp: $(GEN_LAYOUTS)
	$(YK_MK_GEN_DIR)
	$< -d $(max_dims) > $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because indent program failed or was not found."

$(YK_GEN_DIR)/yask_grid_code.hpp: $(GEN_LAYOUTS)
	$(YK_MK_GEN_DIR)
	$< -g $(max_dims) > $@

# Extract the number of stencil dims from the compiler output.
# Use this to create an option to pass to the loop generator script.
$(YK_DIMS_FILE): $(YK_CODE_FILE)
	awk '/NUM_STENCIL_DIMS/ {print "-ndims",$$NF}' $< > $@

$(YK_CODE_FILE): $(YC_EXEC)
	$(YK_MK_GEN_DIR)
	$(RUN_PREFIX) $< $(YC_FLAGS) $(EXTRA_YC_FLAGS) -p $(YC_TARGET) $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "is not properly indented because no indent program was found."

$(YK_MACRO_FILE):
	$(YK_MK_GEN_DIR)
	echo '// Settings from YASK Makefile' > $@
	echo '// Automatically-generated code; do not edit.' >> $@
	for macro in $(MACROS) $(EXTRA_MACROS); do \
	  echo '#define' $$macro | sed 's/=/ /' >> $@; \
	done

headers: $(YK_GEN_HEADERS)
	@ echo 'Header files generated.'

######## API targets
# NB: must set stencil and arch to generate the desired kernel API.

# Build C++ and Python kernel API libs.
api: $(YK_LIB) $(YK_PY_LIB)

# Build python kernel API lib.
# TODO: consider adding $(YK_TAG) to [some of] these targets.
$(YK_SWIG_DIR)/yask_kernel_api_wrap.cpp: $(YK_SWIG_DIR)/yask*.i $(INC_DIR)/*.hpp
	$(SWIG) -version
	$(SWIG) -v -DYK_MODULE=$(YK_MODULE) -cppext cpp \
	  -I$(INC_DIR) -I$(COMM_DIR) -I$(COMM_DIR)/swig \
	  -c++ -python -outdir $(YASK_DIR) -builtin $<

$(YK_SWIG_DIR)/yask_kernel_api_wrap.o: $(YK_SWIG_DIR)/yask_kernel_api_wrap.cpp
	$(YK_CXX) $(YK_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<

$(YK_PY_LIB): $(YK_OBJS) $(YK_SWIG_DIR)/yask_kernel_api_wrap.o
	$(YK_CXX) $(YK_CXXFLAGS) -shared -o $@ $^

# Simple tests

$(YK_GRID_TEST_EXEC): ./tests/grid_test.cpp $(YK_LIB)
	$(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@

$(YK_TUPLE_TEST_EXEC): ./tests/tuple_test.cpp $(YK_LIB)
	$(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@

#### API tests.

# Build C++ kernel tests.
$(YK_API_TEST_EXEC): ./tests/yask_kernel_api_test.cpp $(YK_LIB)
	$(YK_CXX) $(YK_CXXFLAGS) $< $(YK_LFLAGS) -o $@

# Run C++ kernel tests.
cxx-yk-api-test: $(YK_API_TEST_EXEC)
	@echo '*** Running the C++ YASK kernel API test...'
	$(RUN_PREFIX) $<

cxx-yk-grid-test: $(YK_GRID_TEST_EXEC)
	@echo '*** Running the C++ YASK grid test...'
	$(RUN_PREFIX) $<

cxx-yk-tuple-test: $(YK_TUPLE_TEST_EXEC)
	@echo '*** Running the C++ YASK tuple test...'
	$(RUN_PREFIX) $<

# Run Python kernel API test.
py-yk-api-test: $(BIN_DIR)/yask_kernel_api_test.py $(YK_PY_LIB)
	@echo '*** Running the Python YASK kernel API test...'
	$(RUN_PYTHON) $<

### API tests for the compiler.
# These targets run the tests in the compiler directory,
# then they move the output files into the correct location
# for the kernel.

# Run Python compiler API test to create stencil-code file.
py-yc-api-test:
	$(MAKE) -C $(YC_SRC_DIR) $@
	$(YK_MK_GEN_DIR)
	mv $(YC_SRC_DIR)/yc-api-test-py.hpp $(YK_CODE_FILE)

# Run C++ compiler API test to create stencil-code file.
cxx-yc-api-test:
	$(MAKE) -C $(YC_SRC_DIR) $@
	$(YK_MK_GEN_DIR)
	mv $(YC_SRC_DIR)/yc-api-test-cxx.hpp $(YK_CODE_FILE)

######## Misc targets

# Run the default YASK compiler and kernel.
yc-and-yk-test: $(YK_EXEC)
	$(BIN_DIR)/yask.sh -stencil $(stencil) -arch $(arch) -v

# Generate the code file using the built-in compiler.
code-file: $(YK_CODE_FILE)

# Special target to avoid running stencil compiler and replacing the stencil-code file.
# NB: This trick is only needed when using the compiler API to create
# a stencil to replace the one normally created by the pre-built stencil compiler.
NO_YC_MAKE_FLAGS := --new-file=$(YK_CODE_FILE)
kernel-only:
	$(MAKE) $(NO_YC_MAKE_FLAGS)

# Run the YASK kernel test without implicity using the YASK compiler.
yk-test-no-yc: kernel-only
	$(BIN_DIR)/yask.sh -stencil $(stencil) -arch $(arch) -v

# NB: set arch var if applicable.
# NB: save some time by using YK_CXXOPT=-O2.
# These tests are focused on the kernel and not the compiler.
# For testing both the kernel and compiler in various combinations,
# run the tests from the top-level Makefile.
all-tests:
	$(MAKE) clean; $(MAKE) cxx-yk-tuple-test
	$(MAKE) clean; $(MAKE) stencil=3axis fold=x=4,y=2 cxx-yk-grid-test
	$(MAKE) clean; $(MAKE) stencil=iso3dfd fold=x=4,y=2 yc-and-yk-test
	$(MAKE) clean; $(MAKE) stencil=awp_elastic yc-and-yk-test real_bytes=8
	$(MAKE) clean; $(MAKE) stencil=fsg_abc yc-and-yk-test real_bytes=8
	$(MAKE) clean; $(MAKE) stencil=cube yc-and-yk-test real_bytes=8
	$(MAKE) clean; $(MAKE) stencil=iso3dfd cxx-yk-api-test
	$(MAKE) clean; $(MAKE) stencil=iso3dfd py-yk-api-test

all:
	$(MAKE) kernel
	$(MAKE) api
	$(MAKE) all-tests

# Remove intermediate files.
# Should not trigger remake of stencil compiler.
# Make this target before rebuilding YASK with any new parameters.
clean:
	rm -fv *.s
	rm -fv num_dims.*.txt
	rm -fr $(YK_SWIG_DIR)/build $(YK_GEN_DIR)
	rm -fv $(YK_SWIG_DIR)/*_api_wrap.*
	rm -fv $(YK_OBJS)
	find . $(COMM_DIR) -name '*.*.o' | xargs -r rm -v
	find . $(COMM_DIR) $(BIN_DIR) $(LIB_DIR) -name '*.optrpt' | xargs -r rm -v

# Remove executables, libs, etc.
realclean: clean
	rm -fv $(YK_LIB) $(YK_EXEC) $(YK_API_TEST_EXEC) $(YK_PY_MOD)* $(YK_PY_LIB)
	rm -fv make-report.*.txt
	find . -name '*.pyc' | xargs -r rm -v
	find . -name '*~' | xargs -r rm -v

echo-settings:
	@echo
	@echo "Build environment for" $(YK_EXEC) on `date`
	@echo host=`hostname`
	@echo stencil=$(stencil)
	@echo arch=$(arch)
	@echo def_thread_divisor=$(def_thread_divisor)
	@echo def_block_threads=$(def_block_threads)
	@echo def_rank_args=$(def_rank_args)
	@echo def_block_args=$(def_block_args)
	@echo def_pad_args=$(def_pad_args)
	@echo more_def_args=$(more_def_args)
	@echo EXTRA_DEF_ARGS=$(EXTRA_DEF_ARGS)
	@echo fold=$(fold)
	@echo cluster=$(cluster)
	@echo radius=$(radius)
	@echo real_bytes=$(real_bytes)
	@echo pfd_l1=$(pfd_l1)
	@echo pfd_l2=$(pfd_l2)
	@echo streaming_stores=$(streaming_stores)
	@echo omp_region_schedule=$(omp_region_schedule)
	@echo omp_block_schedule=$(omp_block_schedule)
	@echo omp_gen_schedule=$(omp_gen_schedule)
	@echo YC_TARGET="\"$(YC_TARGET)\""
	@echo YC_FLAGS="\"$(YC_FLAGS)\""
	@echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""
	@echo MACROS="\"$(MACROS)\""
	@echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""
	@echo ISA=$(ISA)
	@echo OMPFLAGS="\"$(OMPFLAGS)\""
	@echo EXTRA_YK_CXXFLAGS="\"$(EXTRA_YK_CXXFLAGS)\""
	@echo YK_CXX=$(YK_CXX)
	@$(YK_CXX) --version
	@echo YK_CXXOPT=$(YK_CXXOPT)
	@echo YK_CXXFLAGS="\"$(YK_CXXFLAGS)\""
	@echo RANK_LOOP_OPTS="\"$(RANK_LOOP_OPTS)\""
	@echo RANK_LOOP_OUTER_MODS="\"$(RANK_LOOP_OUTER_MODS)\""
	@echo RANK_LOOP_OUTER_VARS="\"$(RANK_LOOP_OUTER_VARS)\""
	@echo RANK_LOOP_INNER_MODS="\"$(RANK_LOOP_INNER_MODS)\""
	@echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""
	@echo REGION_LOOP_OPTS="\"$(REGION_LOOP_OPTS)\""
	@echo REGION_LOOP_OUTER_MODS="\"$(REGION_LOOP_OUTER_MODS)\""
	@echo REGION_LOOP_OUTER_VARS="\"$(REGION_LOOP_OUTER_VARS)\""
	@echo REGION_LOOP_INNER_MODS="\"$(REGION_LOOP_INNER_MODS)\""
	@echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""
	@echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""
	@echo BLOCK_LOOP_OUTER_MODS="\"$(BLOCK_LOOP_OUTER_MODS)\""
	@echo BLOCK_LOOP_OUTER_VARS="\"$(BLOCK_LOOP_OUTER_VARS)\""
	@echo BLOCK_LOOP_INNER_MODS="\"$(BLOCK_LOOP_INNER_MODS)\""
	@echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""
	@echo SUB_BLOCK_LOOP_OPTS="\"$(SUB_BLOCK_LOOP_OPTS)\""
	@echo SUB_BLOCK_LOOP_OUTER_MODS="\"$(SUB_BLOCK_LOOP_OUTER_MODS)\""
	@echo SUB_BLOCK_LOOP_OUTER_VARS="\"$(SUB_BLOCK_LOOP_OUTER_VARS)\""
	@echo SUB_BLOCK_LOOP_INNER_MODS="\"$(SUB_BLOCK_LOOP_INNER_MODS)\""
	@echo SUB_BLOCK_LOOP_INNER_VARS="\"$(SUB_BLOCK_LOOP_INNER_VARS)\""
	@echo SUB_BLOCK_LOOP_CODE="\"$(SUB_BLOCK_LOOP_CODE)\""
	@echo GEN_LOOP_OPTS="\"$(GEN_LOOP_OPTS)\""
	@echo GEN_LOOP_OUTER_MODS="\"$(GEN_LOOP_OUTER_MODS)\""
	@echo GEN_LOOP_OUTER_VARS="\"$(GEN_LOOP_OUTER_VARS)\""
	@echo GEN_LOOP_INNER_MODS="\"$(GEN_LOOP_INNER_MODS)\""
	@echo GEN_LOOP_CODE="\"$(GEN_LOOP_CODE)\""

code-stats:
	@echo
	@echo "Code stats for stencil computation:"
	$(GET_LOOP_STATS) -t=$(YK_CODE_FILE) factory.s

help:
	@echo "Example performance builds of kernel cmd-line tool:"
	@echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=iso3dfd"
	@echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=awp"
	@echo " $(MAKE) clean; $(MAKE) -j arch=skx stencil=3axis fold='x=1,y=2,z=4' cluster='x=2'"
	@echo " $(MAKE) clean; $(MAKE) -j arch=hsw stencil=3axis radius=4"
	@echo " "
	@echo "Example performance builds of kernel API for C++ and Python apps:"
	@echo " $(MAKE) clean; $(MAKE) -j arch=knl stencil=iso3dfd yk-api"
	@echo " $(MAKE) clean; $(MAKE) -j arch=skx stencil=awp yk-api"
	@echo " "
	@echo "Example debug builds of kernel cmd-line tool:"
	@echo " $(MAKE) clean; $(MAKE) -j stencil=iso3dfd mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' EXTRA_MACROS='DEBUG'"
	@echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=3axis mpi=0 OMPFLAGS='-qopenmp-stubs' YK_CXXOPT='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo " $(MAKE) clean; $(MAKE) -j arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' mpi=0 YK_CXX=g++ OMPFLAGS='' YK_CXXOPT='-O0' EXTRA_MACROS='DEBUG TRACE TRACE_MEM TRACE_INTRINSICS'"
	@echo " "
	@echo "Example builds with testing:"
	@echo " $(MAKE) -j all"
	@echo " $(MAKE) -j all YK_CXX=g++ YK_CXXOPT=-O2 mpi=0"
