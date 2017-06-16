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

# Type 'make help' for some examples.
#
# Some of the make vars that are commonly set via the command-line:
#
# stencil: see list below.
#
# arch: see list below.
#
# mpi: 0, 1: whether to use MPI. 
#
# radius: sets size of certain stencils.
#
# real_bytes: FP precision: 4=float, 8=double.
#
# fold: How to fold vectors (x*y*z).
#   Vectorization in dimensions perpendicular to the inner loop
#   (defined by SUB_BLOCK_LOOP_INNER_VARS below) often works well.
# fold_4byte: How to fold vectors when real_bytes=4.
# fold_8byte: How to fold vectors when real_bytes=8.
#
# cluster: How many folded vectors to evaluate simultaneously.
#
# eqs: comma-separated name=substr pairs used to group
#   grid update equations into sets.
#
# streaming_stores: 0, 1: Whether to use streaming stores.
#
# hbw: 0, 1: whether to use memkind lib.
#   If hbw=1, the memkind lib will be used to allocate grids;
#   this can provide the ability to fine-tune which grids use
#   HBW and which use default memory.
#
# pfd_l1: L1 prefetch distance (only if enabled in sub-block loop).
# pfd_l2: L2 prefetch distance (only if enabled in sub-block loop).
#
# omp_region_schedule: OMP schedule policy for region loop.
# omp_block_schedule: OMP schedule policy for nested OpenMP block loop.
# omp_halo_schedule: OMP schedule policy for OpenMP halo loop.
#
# def_block_threads: Number of threads to use in nested OpenMP block loop by default.
# def_thread_divisor: Divide number of OpenMP threads by this factor by default.
# def_*_args: Default cmd-line args for specific settings.
# more_def_args: Additional default cmd-line args.

# Convention: when useful for distinction,
# - vars starting with 'YK_' apply to the run-time stencil kernel.
# - vars starting with 'YC_' apply to the stencil compiler.

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
  SUB_BLOCK_LOOP_INNER_MODS	=
 else ifeq ($(arch),skx)
  fold_4byte			=	x=4,y=4
  def_thread_divisor		=	1
  def_block_threads		=	2
  def_block_args		=	-b 64
  cluster			=	z=2
  SUB_BLOCK_LOOP_INNER_MODS	=	prefetch(L1)
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
  SUB_BLOCK_LOOP_INNER_MODS	=	prefetch(L1,L2)
  omp_block_schedule		=	dynamic,1
  fold_4byte			=	x=8
  cluster			=	y=2
  def_pad_args			=	-epx 1 -epy 1 -epz 0
  def_block_args		=	-bx 128 -by 16 -bz 32 
  more_def_args			+=	-sbx 32 -sby 2 -sbz 32
 else ifeq ($(arch),skx)
  fold_4byte			=	x=2,y=8
  def_block_args		=	-b 32 -bx 96
  SUB_BLOCK_LOOP_INNER_MODS	=	prefetch(L1)
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
  SUB_BLOCK_LOOP_INNER_MODS  =	prefetch(L2)
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
 SUB_BLOCK_LOOP_INNER_MODS  ?=	prefetch(L1,L2)

else ifeq ($(arch),knl)

 ISA		?=	-xMIC-AVX512
 GCXX_ISA	?=	-march=knl
 MACROS		+=	USE_INTRIN512 USE_RCP28
 YC_TARGET  	?=	avx512
 def_block_args	?=	-b 96
 def_block_threads ?=	8
 SUB_BLOCK_LOOP_INNER_MODS  ?=	prefetch(L1)

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
omp_halo_schedule	?=	static
def_thread_divisor	?=	1
def_block_threads	?=	2
real_bytes		?=	4
layout_xyz		?=	Layout_123
layout_txyz		?=	Layout_2314
def_rank_args		?=	-d 128
def_block_args		?=	-b 64
def_pad_args		?=	-ep 1
cluster			?=	x=1
pfd_l1			?=	1
pfd_l2			?=	2

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

# Flags for stencil compiler.
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

# Compiler, etc.
ifeq ($(mpi),1)
 CXX		:=	mpiicpc
else
 CXX		:=	icpc
endif
LD		:=	$(CXX)
PYTHON		:=	python

# More build flags.
CXXOPT		?=	-O3
CXXFLAGS        +=   	-g -std=c++11 -Wall $(CXXOPT)
CXXFLAGS	+=	-Iinclude -Isrc/common -Isrc/kernel/lib -Isrc/kernel/gen
OMPFLAGS	+=	-fopenmp

# Find include path needed for python interface.
# NB: constructing string inside print() to work for python 2 or 3.
PYINC		:= 	$(addprefix -I,$(shell $(PYTHON) -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc() + " " + distutils.sysconfig.get_python_inc(plat_specific=1))'))

CWD		:=	$(shell pwd)
LFLAGS          +=      -lrt -Wl,-rpath=$(CWD)/lib -L$(CWD)/lib
YC_LFLAGS	:=	$(LFLAGS)
YK_LFLAGS	:=	$(LFLAGS)

YC_CXX    	?=	g++  # usually faster than icpc for building the compiler.
YC_CXXFLAGS 	+=	-g -std=c++11 -Wall -O2
YC_CXXFLAGS	+=	-Iinclude -Isrc/common -Isrc/compiler/lib -Isrc/stencils

# Default cmd-line args.
DEF_ARGS	+=	-thread_divisor $(def_thread_divisor)
DEF_ARGS	+=	-block_threads $(def_block_threads)
DEF_ARGS	+=	$(def_rank_args) $(def_block_args) $(def_pad_args) $(more_def_args) 
MACROS		+=	DEF_ARGS='"$(DEF_ARGS) $(EXTRA_DEF_ARGS)"'

# Set more MACROS based on individual makefile vars.
# MACROS and EXTRA_MACROS will be written to a header file.
MACROS		+=	LAYOUT_XYZ=$(layout_xyz)
MACROS		+=	LAYOUT_TXYZ=$(layout_txyz)
MACROS		+=	PFDL1=$(pfd_l1) PFDL2=$(pfd_l2)
ifeq ($(streaming_stores),1)
 MACROS		+=	USE_STREAMING_STORE
endif

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
MACROS		+= 	ARCH_$(ARCH)

# MPI settings.
ifeq ($(mpi),1)
 MACROS		+=	USE_MPI
endif

# HBW settings.
ifeq ($(hbw),1)
 MACROS		+=	USE_HBW
 HBW_DIR 	=	$(HOME)/memkind_build
 CXXFLAGS	+=	-I$(HBW_DIR)/include
 YK_LFLAGS	+= 	-lnuma $(HBW_DIR)/lib/libmemkind.a
endif

# VTUNE settings.
ifeq ($(vtune),1)
 MACROS		+=	USE_VTUNE
ifneq ($(VTUNE_AMPLIFIER_XE_2017_DIR),)
 VTUNE_DIR	=	$(VTUNE_AMPLIFIER_XE_2017_DIR)
else
 VTUNE_DIR	=	$(VTUNE_AMPLIFIER_XE_2016_DIR)
endif
CXXFLAGS	+=	-I$(VTUNE_DIR)/include
YK_LFLAGS	+=	$(VTUNE_DIR)/lib64/libittnotify.a
endif

# compiler-specific settings
ifneq ($(findstring ic,$(notdir $(CXX))),)  # Intel compiler

 CXXFLAGS	+=      $(ISA) -debug extended -Fa -restrict -ansi-alias -fno-alias
 CXXFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt \
			-no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz \
			-fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 #CXXFLAGS	+=	-qoverride-limits
 CXXFLAGS	+=	-vec-threshold0
 CXXFLAGS	+=      -qopt-report=5
 #CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 CXXFLAGS	+=	-no-diag-message-catalog
 CXX_VER_CMD	:=	$(CXX) -V

 # work around an optimization bug.
 MACROS		+=	NO_STORE_INTRINSICS

else # not Intel compiler
 CXXFLAGS	+=	$(GCXX_ISA) -Wno-unknown-pragmas -Wno-unused-variable

endif # compiler.

# Compile with model_cache=1 or 2 to check prefetching.
ifeq ($(model_cache),1)
 MACROS       	+=      MODEL_CACHE=1
 OMPFLAGS	=	-qopenmp-stubs
else ifeq ($(model_cache),2)
 MACROS       	+=      MODEL_CACHE=2
 OMPFLAGS	=	-qopenmp-stubs
endif

# Add in OMP flags and user-added flags.
CXXFLAGS	+=	$(OMPFLAGS) $(EXTRA_CXXFLAGS)
YC_CXXFLAGS	+=	$(EXTRA_YC_CXXFLAGS)

# More file names.
YC_SRC_BASES	:=	CppIntrin Expr ExprUtils Print
YC_OBJS		:=	$(addprefix src/compiler/lib/,$(addsuffix .o,$(YC_SRC_BASES)))
YC_BASE		:=	yask_compiler
YC_LIB		:=	lib/lib$(YC_BASE).so
YC_EXEC		:=	bin/$(YC_BASE).exe
YC_STENCIL_LIST	:=	src/compiler/stencils.hpp
YC_SWIG_DIR	:=	src/compiler/swig

YK_GEN_DIR	:=	src/kernel/gen
YK_SWIG_DIR	:=	src/kernel/swig
YK_MACRO_FILE	:=	$(YK_GEN_DIR)/yask_macros.hpp
YK_CODE_FILE	:=	$(YK_GEN_DIR)/yask_stencil_code.hpp
YK_GEN_HEADERS	:=	$(addprefix $(YK_GEN_DIR)/, \
				yask_rank_loops.hpp \
				yask_region_loops.hpp \
				yask_block_loops.hpp \
				yask_sub_block_loops.hpp \
				yask_halo_loops.hpp \
				yask_layout_macros.hpp \
				yask_layouts.hpp) \
				$(YK_MACRO_FILE) \
				$(YK_CODE_FILE)

YK_TAG		:=	$(stencil).$(arch)
YK_SRC_BASES	:=	stencil_calc realv_grids utils
YK_OBJS		:=	$(addprefix src/kernel/lib/,$(addsuffix .$(YK_TAG).o,$(YK_SRC_BASES)))
YK_BASE		:=	yask_kernel.$(YK_TAG)
YK_LIB		:=	lib/lib$(YK_BASE).so
YK_EXEC		:=	bin/$(YK_BASE).exe

MAKE_REPORT_FILE:=	make-report.$(YK_TAG).txt
CXXFLAGS_FILE	:=	cxx-flags.$(YK_TAG).txt
LFLAGS_FILE	:=	ld-flags.$(YK_TAG).txt

YK_MK_GEN_DIR	:=	mkdir -p -v $(YK_GEN_DIR)

# gen_loops.pl args:

# Rank loops break up the whole rank into smaller regions.  In order for
# temporal wavefronts to operate properly, the order of spatial dimensions
# may be changed, but the scanning paths must have strictly incrementing
# indices. Those that do not (e.g., grouped, serpentine, square-wave) may
# *not* be used here when using temporal wavefronts. The time loop may be
# found in StencilEquations::calc_rank().
RANK_LOOP_OPTS		?=	-dims 'dx,dy,dz'
RANK_LOOP_OUTER_VARS	?=	dx,dy,dz
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop($(RANK_LOOP_OUTER_VARS)) \
				{ $(RANK_LOOP_INNER_MODS) calc(region(start_dt, stop_dt, eqGroup_ptr)); }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.  The region time loops are not coded here to
# allow for proper spatial skewing for temporal wavefronts. The time loop
# may be found in StencilEquations::calc_region().
REGION_LOOP_OPTS	?=     	-dims 'rx,ry,rz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_region_schedule)) proc_bind(spread)' \
				-calcPrefix 'eg->calc_'
REGION_LOOP_OUTER_VARS	?=	rx,ry,rz
REGION_LOOP_OUTER_MODS	?=	grouped
REGION_LOOP_CODE	?=	omp $(REGION_LOOP_OUTER_MODS) loop($(REGION_LOOP_OUTER_VARS)) { \
				$(REGION_LOOP_INNER_MODS) calc(block(rt)); }

# Block loops break up a block into sub-blocks.  The 'omp' modifier creates
# a nested OpenMP loop so that each sub-block is assigned to a nested OpenMP
# thread.  There is no time loop here because threaded temporal blocking is
# not yet supported.
BLOCK_LOOP_OPTS		=     	-dims 'bx,by,bz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)'
BLOCK_LOOP_OUTER_VARS	?=	bx,by,bz
BLOCK_LOOP_OUTER_MODS	?=	grouped
BLOCK_LOOP_CODE		?=	omp $(BLOCK_LOOP_OUTER_MODS) loop($(BLOCK_LOOP_OUTER_VARS)) { \
				$(BLOCK_LOOP_INNER_MODS) calc(sub_block(bt)); }

# Sub-block loops break up a sub-block into vector clusters.  The indices at
# this level are by vector instead of element; this is indicated by the 'v'
# suffix. The innermost loop here is the final innermost loop. There is
# no time loop here because threaded temporal blocking is not yet supported.
SUB_BLOCK_LOOP_OPTS		=     	-dims 'sbxv,sbyv,sbzv'
ifeq ($(split_L2),1)
 SUB_BLOCK_LOOP_OPTS		+=     	-splitL2
endif
SUB_BLOCK_LOOP_OUTER_VARS	?=	sbxv,sbyv
SUB_BLOCK_LOOP_OUTER_MODS	?=	square_wave serpentine
SUB_BLOCK_LOOP_INNER_VARS	?=	sbzv
SUB_BLOCK_LOOP_INNER_MODS	?=	prefetch(L2)
SUB_BLOCK_LOOP_CODE		?=	$(SUB_BLOCK_LOOP_OUTER_MODS) loop($(SUB_BLOCK_LOOP_OUTER_VARS)) { \
					$(SUB_BLOCK_LOOP_INNER_MODS) loop($(SUB_BLOCK_LOOP_INNER_VARS)) { \
					calc(cluster(begin_sbtv)); } }

# Halo pack/unpack loops break up a region face, edge, or corner into vectors.
# Nested OpenMP is not used here because there is no sharing between threads.
# TODO: Consider using nested OpenMP to hide more latency.
HALO_LOOP_OPTS		=     	-dims 'hx,hy,hz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_halo_schedule)) proc_bind(spread)'
HALO_LOOP_OUTER_MODS	?=	omp
HALO_LOOP_OUTER_VARS	?=	hx,hy,hz
HALO_LOOP_CODE		?=	$(HALO_LOOP_OUTER_MODS) loop($(HALO_LOOP_OUTER_VARS)) \
				$(HALO_LOOP_INNER_MODS) { calc(halo(t)); }

######## Primary targets & rules
# NB: must set stencil and arch to generate the YASK kernel.

all:	$(YK_EXEC) $(MAKE_REPORT_FILE)
	echo $(CXXFLAGS) > $(CXXFLAGS_FILE)
	echo $(YK_LFLAGS) > $(LFLAGS_FILE)
	@cat $(MAKE_REPORT_FILE)
	@echo "Binary" $(YK_EXEC) "has been built."
	@echo "Run command: bin/yask.sh -stencil" $(stencil) "-arch" $(arch) "[options]"

$(MAKE_REPORT_FILE): $(YK_EXEC)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	$(MAKE) echo-settings >> $@ 2>&1
	$(MAKE) code-stats >> $@ 2>&1

# Compile the kernel.
%.$(YK_TAG).o: %.cpp src/common/*hpp src/kernel/lib/*hpp include/*hpp $(YK_GEN_HEADERS)
	$(CXX) --version
	$(CXX) $(CXXFLAGS) -fPIC -c -o $@ $<

$(YK_LIB): $(YK_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

$(YK_EXEC): src/kernel/yask_main.cpp $(YK_LIB)
	$(CXX) $(CXXFLAGS) $(YK_LFLAGS) -o $@ $< -l$(YK_BASE)

kernel: $(YK_EXEC)

yk-test: $(YK_EXEC)
	bin/yask.sh -stencil $(stencil) -arch $(arch) -v

# Auto-generated files.
$(YK_GEN_DIR)/yask_rank_loops.hpp: bin/gen_loops.pl Makefile
	$(YK_MK_GEN_DIR)
	$< -output $@ $(RANK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_RANK_LOOP_OPTS) "$(RANK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_region_loops.hpp: bin/gen_loops.pl Makefile
	$(YK_MK_GEN_DIR)
	$< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

$(YK_GEN_DIR)/yask_block_loops.hpp: bin/gen_loops.pl Makefile
	$(YK_MK_GEN_DIR)
	$< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_sub_block_loops.hpp: bin/gen_loops.pl Makefile
	$(YK_MK_GEN_DIR)
	$< -output $@ $(SUB_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_SUB_BLOCK_LOOP_OPTS) "$(SUB_BLOCK_LOOP_CODE)"

$(YK_GEN_DIR)/yask_halo_loops.hpp: bin/gen_loops.pl Makefile
	$(YK_MK_GEN_DIR)
	$< -output $@ $(HALO_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_HALO_LOOP_OPTS) "$(HALO_LOOP_CODE)"

$(YK_GEN_DIR)/yask_layout_macros.hpp: bin/gen_layouts.pl
	$(YK_MK_GEN_DIR)
	$< -m > $@

$(YK_GEN_DIR)/yask_layouts.hpp: bin/gen_layouts.pl
	$(YK_MK_GEN_DIR)
	$< -d > $@

$(YK_CODE_FILE): $(YC_EXEC)
	$(YK_MK_GEN_DIR)
	$< $(YC_FLAGS) $(EXTRA_YC_FLAGS) -p $(YC_TARGET) $@
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

headers: $(GEN_HEADERS)
	@ echo 'Header files generated.'

# Compile the stencil compiler.
%.o: %.cpp src/common/*hpp src/compiler/lib/*hpp include/yask_compiler_api.hpp
	$(YC_CXX) --version
	$(YC_CXX) $(YC_CXXFLAGS) -fPIC -c -o $@ $<

$(YC_LIB): $(YC_OBJS)
	$(YC_CXX) $(YC_CXXFLAGS) -shared -o $@ $^

$(YC_EXEC): src/compiler/main.cpp $(YC_LIB) $(YC_STENCIL_LIST)
	$(YC_CXX) $(YC_CXXFLAGS) -O0 $(YC_LFLAGS) -o $@ $< -l$(YC_BASE)

$(YC_STENCIL_LIST): src/stencils/*.hpp
	echo '// Stencil-definition files.' > $@
	echo '// Automatically-generated code; do not edit.' >> $@
	for sfile in $(^F); do \
	  echo '#include "'$$sfile'"' >> $@; \
	done

compiler: $(YC_EXEC)

######## API targets
# NB: must set stencil and arch to generate the kernel API.

# API docs & libs.
api-all: api-docs yc-api yk-api

# Format API documents.
api-docs: docs/api/html/index.html

# Build C++ and Python compiler API libs.
yc-api: $(YC_LIB) lib/_yask_compiler.so

# Build C++ and Python kernel API libs.
yk-api: $(YK_LIB) lib/_yask_kernel.so


# Format API documents.
docs/api/html/index.html: include/*.hpp docs/api/*.*
	doxygen -v
	cd docs/api; doxygen doxygen_config.txt
	@ echo Open $@ 'in a browser to view the API docs.'

# Build python compiler API lib.
$(YC_SWIG_DIR)/yask_compiler_api_wrap.cpp: $(YC_SWIG_DIR)/yask*.i include/*hpp
	swig -version
	swig -v -cppext cpp -Iinclude -c++ -python -outdir lib -builtin $<

$(YC_SWIG_DIR)/yask_compiler_api_wrap.o: $(YC_SWIG_DIR)/yask_compiler_api_wrap.cpp
	$(YC_CXX) $(YC_CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<

lib/_yask_compiler.so: $(YC_OBJS) $(YC_SWIG_DIR)/yask_compiler_api_wrap.o
	$(YC_CXX) $(YC_CXXFLAGS) -shared -o $@ $^

# Build python kernel API lib.
# TODO: consider adding $(YK_TAG) to [some of] these targets.
$(YK_SWIG_DIR)/yask_kernel_api_wrap.cpp: $(YK_SWIG_DIR)/yask*.i include/*hpp
	swig -version
	swig -v -cppext cpp -Iinclude -c++ -python -outdir lib -builtin $<

$(YK_SWIG_DIR)/yask_kernel_api_wrap.o: $(YK_SWIG_DIR)/yask_kernel_api_wrap.cpp
	$(CXX) $(CXXFLAGS) $(PYINC) -fPIC -c -o $@ $<

lib/_yask_kernel.so: $(YK_OBJS) $(YK_SWIG_DIR)/yask_kernel_api_wrap.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

# Build C++ compiler API test.
bin/yask_compiler_api_test.exe: $(YC_LIB) src/compiler/tests/yask_compiler_api_test.cpp
	$(YC_CXX) $(YC_CXXFLAGS) -o $@ $^

# Build C++ kernel API test.
bin/yask_kernel_api_test.exe: $(YK_LIB) src/kernel/tests/yask_kernel_api_test.cpp
	$(CXX) $(CXXFLAGS) $(LFLAGS) -o $@ $^

# Special target to avoid running stencil compiler and replacing the stencil-code file.
# NB: This trick is only needed when using the compiler API to create
# a stencil to replace the one normally created by the pre-built stencil compiler.
NO_YC_MAKE_FLAGS := --new-file=$(YK_CODE_FILE)
kernel-only:
	$(MAKE) $(NO_YC_MAKE_FLAGS)

# Run Python compiler API test to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
py-yc-api-test: bin/yask_compiler_api_test.py lib/_yask_compiler.so
	$(MAKE) clean
	@echo '*** Running the Python YASK compiler API test...'
	$(PYTHON) $<
	$(YK_MK_GEN_DIR)
	mv yc-api-test-py.hpp $(YK_CODE_FILE)
	- dot -Tpdf -O yc-api-test-py.dot && ls -l yc-api-test-py.dot*

# Run C++ compiler API test to create stencil-code file.
# Also create .pdf rendering of stencil AST if Graphviz is installed.
cxx-yc-api-test: bin/yask_compiler_api_test.exe
	$(MAKE) clean
	@echo '*** Running the C++ YASK compiler API test...'
	$<
	$(YK_MK_GEN_DIR)
	mv yc-api-test-cxx.hpp $(YK_CODE_FILE)
	- dot -Tpdf -O yc-api-test-cxx.dot && ls -l yc-api-test-cxx.dot*

# Run the YASK kernel without implicity using the YASK compiler.
yk-test-no-yc: kernel-only
	bin/yask.sh -stencil $(stencil) -arch $(arch) -v

# Run C++ compiler API test, then run YASK kernel using its output.
cxx-yc-api-and-yk-test: cxx-yc-api-test
	$(MAKE) yk-test-no-yc

# Run Python compiler API test, then run YASK kernel using its output.
py-yc-api-and-yk-test: py-yc-api-test
	$(MAKE) yk-test-no-yc

# Run C++ kernel API test.
cxx-yk-api-test: bin/yask_kernel_api_test.exe
	@echo '*** Running the C++ YASK kernel API test...'
	$<

# Run Python kernel API test.
py-yk-api-test: bin/yask_kernel_api_test.py lib/_yask_kernel.so
	@echo '*** Running the Python YASK kernel API test...'
	$(PYTHON) $<

# Run C++ compiler API test, then run C++ kernel API test.
cxx-yc-api-and-cxx-yk-api-test: cxx-yc-api-test
	$(MAKE) $(NO_YC_MAKE_FLAGS) cxx-yk-api-test

# Run C++ compiler API test, then run python kernel API test.
cxx-yc-api-and-py-yk-api-test: cxx-yc-api-test
	$(MAKE) $(NO_YC_MAKE_FLAGS) py-yk-api-test

# Run python compiler API test, then run C++ kernel API test.
py-yc-api-and-cxx-yk-api-test: py-yc-api-test
	$(MAKE) $(NO_YC_MAKE_FLAGS) cxx-yk-api-test

# Run python compiler API test, then run python kernel API test.
py-yc-api-and-py-yk-api-test: py-yc-api-test
	$(MAKE) $(NO_YC_MAKE_FLAGS) py-yk-api-test


######## Misc targets

# NB: set arch var if applicable.
# NB: save some time by using CXXOPT=-O2.
all-tests:
	$(MAKE) clean; $(MAKE) -j stencil=iso3dfd yk-test
	$(MAKE) clean; $(MAKE) -j stencil=iso3dfd cxx-yk-api-test
	$(MAKE) clean; $(MAKE) -j stencil=iso3dfd py-yk-api-test
	$(MAKE) clean; $(MAKE) -j stencil=fsg_abc yk-test
	$(MAKE) clean; $(MAKE) -j stencil=test cxx-yc-api-and-yk-test
	$(MAKE) clean; $(MAKE) -j stencil=test cxx-yc-api-and-cxx-yk-api-test
	$(MAKE) clean; $(MAKE) -j stencil=test cxx-yc-api-and-py-yk-api-test
	$(MAKE) clean; $(MAKE) -j stencil=test py-yc-api-and-yk-test
	$(MAKE) clean; $(MAKE) -j stencil=test py-yc-api-and-cxx-yk-api-test
	$(MAKE) clean; $(MAKE) -j stencil=test py-yc-api-and-py-yk-api-test

tags:
	rm -f TAGS ; find src include -name '*.[ch]pp' | xargs etags -C -a

# Remove intermediate files.
# Should not trigger remake of stencil compiler.
# Make this target before rebuilding YASK with any new parameters.
clean:
	rm -fv *.s $(MAKE_REPORT_FILE)
	rm -fr src/*/swig/build $(YK_GEN_DIR)
	rm -fv $(YK_SWIG_DIR)/yask_kernel_api_wrap.{cpp,o}
	find src/kernel -name '*.o' | xargs -r rm -v
	find src/kernel -name '*.optrpt' | xargs -r rm -v

# Remove files from old versions.
clean-old:
	rm -fv stencil*.exe stencil-tuner-summary.csh stencil-tuner.pl gen-layouts.pl gen-loops.pl get-loop-stats.pl
	rm -fr docs/html docs/latex
	rm -fv src/tests/*.dot*
	rm -fv src/foldBuilder/*pp

# Remove executables, documentation, etc. (not logs).
realclean: clean clean-old
	rm -fv bin/*.exe lib/*.so make-report*.txt cxx-flags*.txt ld-flags.*txt TAGS $(YC_STENCIL_LIST)
	rm -fr docs/*/html docs/*/latex
	rm -fv $(YC_SWIG_DIR)/yask_compiler_api_wrap.{cpp,o} lib/yask_{compiler,kernel}.py*
	rm -fv *api-test*.dot* *~
	find * -name '*.o' | xargs -r rm -v
	find * -name '*.optrpt' | xargs -r rm -v
	find * -name '*~' | xargs -r rm -v

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
	@echo layout_xyz=$(layout_xyz)
	@echo layout_txyz=$(layout_txyz)
	@echo pfd_l1=$(pfd_l1)
	@echo pfd_l2=$(pfd_l2)
	@echo streaming_stores=$(streaming_stores)
	@echo omp_region_schedule=$(omp_region_schedule)
	@echo omp_block_schedule=$(omp_block_schedule)
	@echo omp_halo_schedule=$(omp_halo_schedule)
	@echo YC_TARGET="\"$(YC_TARGET)\""
	@echo YC_FLAGS="\"$(YC_FLAGS)\""
	@echo EXTRA_YC_FLAGS="\"$(EXTRA_YC_FLAGS)\""
	@echo MACROS="\"$(MACROS)\""
	@echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""
	@echo ISA=$(ISA)
	@echo OMPFLAGS="\"$(OMPFLAGS)\""
	@echo EXTRA_CXXFLAGS="\"$(EXTRA_CXXFLAGS)\""
	@echo CXX=$(CXX)
	@$(CXX) -v; $(CXX_VER_CMD)
	@echo CXXOPT=$(CXXOPT)
	@echo CXXFLAGS="\"$(CXXFLAGS)\""
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
	@echo HALO_LOOP_OPTS="\"$(HALO_LOOP_OPTS)\""
	@echo HALO_LOOP_OUTER_MODS="\"$(HALO_LOOP_OUTER_MODS)\""
	@echo HALO_LOOP_OUTER_VARS="\"$(HALO_LOOP_OUTER_VARS)\""
	@echo HALO_LOOP_INNER_MODS="\"$(HALO_LOOP_INNER_MODS)\""
	@echo HALO_LOOP_CODE="\"$(HALO_LOOP_CODE)\""

code-stats:
	@echo
	@echo "Code stats for stencil computation:"
	bin/get_loop_stats.pl -t='sub_block_loops' *.s

help:
	@echo "Example performance builds of kernel cmd-line tool:"
	@echo "make clean; make -j arch=knl stencil=iso3dfd"
	@echo "make clean; make -j arch=knl stencil=awp"
	@echo "make clean; make -j arch=skx stencil=3axis fold='x=1,y=2,z=4' cluster='x=2'"
	@echo "make clean; make -j arch=hsw stencil=3axis radius=4 SUB_BLOCK_LOOP_INNER_MODS='prefetch(L1,L2)' pfd_l2=3"
	@echo " "
	@echo "Example performance builds of kernel API for C++ and Python apps:"
	@echo "make clean; make -j arch=knl stencil=iso3dfd yk-api"
	@echo "make clean; make -j arch=skx stencil=awp yk-api"
	@echo " "
	@echo "API document generation:"
	@echo "make api-docs   # then see docs/api/html/index.html"
	@echo " "
	@echo "Example debug builds of kernel cmd-line tool:"
	@echo "make clean; make -j stencil=iso3dfd mpi=0 OMPFLAGS='-qopenmp-stubs' CXXOPT='-O0' EXTRA_MACROS='DEBUG'"
	@echo "make clean; make -j arch=intel64 stencil=3axis mpi=0 OMPFLAGS='-qopenmp-stubs' CXXOPT='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo "make clean; make -j arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' mpi=0 OMPFLAGS='-qopenmp-stubs' CXXOPT='-O0' EXTRA_MACROS='DEBUG TRACE TRACE_MEM TRACE_INTRINSICS'"
