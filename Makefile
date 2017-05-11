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

# Some of the make vars available:
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

# Initial defaults.
stencil		=	unspecified
arch		=	knl
mpi		=	0
real_bytes	=	4

# Defaults based on stencil type (and arch for some stencils).
ifeq ($(stencil),)
 $(error Stencil not specified)

else ifeq ($(stencil),ave)
 radius		=	1

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
 def_pad_args			=	-p 1
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
 FB_FLAGS			+=	-min-es 1
 def_rank_args			=	-dx 512 -dy 1024 -dz 128 # assume 2 ranks/node in 'x'.
 def_pad_args			=	-p 1
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
  def_pad_args			=	-px 1 -py 1 -pz 0
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
 FB_TARGET  	?=	knc
 def_block_threads  ?=	4
 SUB_BLOCK_LOOP_INNER_MODS  ?=	prefetch(L1,L2)

else ifeq ($(arch),knl)

 ISA		?=	-xMIC-AVX512
 GCXX_ISA	?=	-march=knl
 MACROS		+=	USE_INTRIN512 USE_RCP28
 FB_TARGET  	?=	avx512
 def_block_args	?=	-b 96
 def_block_threads ?=	8
 SUB_BLOCK_LOOP_INNER_MODS  ?=	prefetch(L1)

else ifeq ($(arch),skx)

 ISA		?=	-xCORE-AVX512
 GCXX_ISA	?=	-march=knl -mno-avx512er -mno-avx512pf
 MACROS		+=	USE_INTRIN512
 FB_TARGET  	?=	avx512
 mpi		=	1

else ifeq ($(arch),hsw)

 ISA		?=	-xCORE-AVX2
 GCXX_ISA	?=	-march=haswell
 MACROS		+=	USE_INTRIN256
 FB_TARGET  	?=	avx2
 mpi		=	1

else ifeq ($(arch),ivb)

 ISA		?=	-xCORE-AVX-I
 GCXX_ISA	?=	-march=ivybridge
 MACROS		+=	USE_INTRIN256
 FB_TARGET  	?=	avx
 mpi		=	1

else ifeq ($(arch),snb)

 ISA		?=	-xAVX
 GCXX_ISA	?=	-march=sandybridge
 MACROS		+= 	USE_INTRIN256
 FB_TARGET  	?=	avx
 mpi		=	1

else ifeq ($(arch),intel64)

 ISA		?=	-xHOST
 GCXX_ISA       ?=      -march=native
 FB_TARGET	?=	cpp

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
layout_wxyz		?=	Layout_1234
layout_twxyz		?=	Layout_23415
def_rank_args		?=	-d 128
def_block_args		?=	-b 64
def_pad_args		?=	-p 0
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

# select fold based on size of reals.
fold	= 	$(fold_$(real_bytes)byte) # e.g., fold_4byte

# More build flags.
ifeq ($(mpi),1)
 CXX		:=	mpiicpc
else
 CXX		:=	icpc
endif
LD		:=	$(CXX)
MAKE		:=	make
CXXOPT		?=	-O3
CXXFLAGS        +=   	-g -std=c++11 -Wall $(CXXOPT)
OMPFLAGS	+=	-fopenmp 
LFLAGS          +=      -lrt
FB_EXEC		:=	bin/foldBuilder.exe
FB_CXX    	?=	g++  # faster than icpc for the foldBuilder.
FB_CXXFLAGS 	+=	-g -O0 -std=c++11 -Wall  # low opt to reduce compile time.
FB_CXXFLAGS	+=	-Iinclude -Isrc/foldBuilder -Isrc/foldBuilder/stencils
FB_FLAGS   	+=	-st $(stencil) -cluster $(cluster) -fold $(fold)
FB_STENCIL_LIST	:=	src/foldBuilder/stencils.hpp
ST_MACRO_FILE	:=	src/stencil_macros.hpp
ST_CODE_FILE	:=	src/stencil_code.hpp
GEN_HEADERS     :=	$(addprefix src/, \
				stencil_rank_loops.hpp \
				stencil_region_loops.hpp \
				stencil_block_loops.hpp \
				stencil_sub_block_loops.hpp \
				stencil_halo_loops.hpp \
				layout_macros.hpp layouts.hpp) \
				$(ST_MACRO_FILE) $(ST_CODE_FILE)
ifneq ($(eqs),)
 FB_FLAGS   	+=	-eq $(eqs)
endif
ifneq ($(radius),)
 FB_FLAGS   	+=	-r $(radius)
endif
ifneq ($(halo),)
 FB_FLAGS   	+=	-halo $(halo)
endif
ifneq ($(time_alloc),)
 FB_FLAGS   	+=	-step-alloc $(time_alloc)
endif

# Default cmd-line args.
DEF_ARGS	+=	-thread_divisor $(def_thread_divisor)
DEF_ARGS	+=	-block_threads $(def_block_threads)
DEF_ARGS	+=	$(def_rank_args) $(def_block_args) $(def_pad_args)
MACROS		+=	DEF_ARGS='"$(DEF_ARGS) $(more_def_args) $(EXTRA_DEF_ARGS)"'

# Set more MACROS based on individual makefile vars.
# MACROS and EXTRA_MACROS will be written to a header file.
MACROS		+=	REAL_BYTES=$(real_bytes)
MACROS		+=	LAYOUT_XYZ=$(layout_xyz)
MACROS		+=	LAYOUT_TXYZ=$(layout_txyz)
MACROS		+=	LAYOUT_WXYZ=$(layout_wxyz)
MACROS		+=	LAYOUT_TWXYZ=$(layout_twxyz)
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
 LFLAGS		+= 	-lnuma $(HBW_DIR)/lib/libmemkind.a
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
LFLAGS		+=	$(VTUNE_DIR)/lib64/libittnotify.a
endif

# compiler-specific settings
ifneq ($(findstring ic,$(notdir $(CXX))),)  # Intel compiler

 CODE_STATS      =   	code_stats
 CXXFLAGS        +=      $(ISA) -debug extended -Fa -restrict -ansi-alias -fno-alias
 CXXFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz -fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
 #CXXFLAGS	+=	-qoverride-limits
 CXXFLAGS	+=	-vec-threshold0
 CXXFLAGS	+=      -qopt-report=5
 #CXXFLAGS	+=	-qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
 CXXFLAGS	+=	-no-diag-message-catalog
 CXX_VER_CMD	=	$(CXX) -V

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

# Some file names.
TAG			:=	$(stencil).$(arch)
STENCIL_BASES		:=	stencil_main stencil_calc realv_grids utils
STENCIL_OBJS		:=	$(addprefix src/,$(addsuffix .$(TAG).o,$(STENCIL_BASES)))
STENCIL_CXX		:=	$(addprefix src/,$(addsuffix .$(TAG).i,$(STENCIL_BASES)))
EXEC_NAME		:=	bin/yask.$(TAG).exe
MAKE_REPORT_FILE	:=	make-report.$(TAG).txt
CXXFLAGS_FILE		:=	cxx-flags.$(TAG).txt
LFLAGS_FILE		:=	ld-flags.$(TAG).txt

# gen-loops.pl args:

# Rank loops break up the whole rank into smaller regions.  In order for
# temporal wavefronts to operate properly, the order of spatial dimensions
# may be changed, but the scanning paths must have strictly incrementing
# indices. Those that do not (e.g., grouped, serpentine, square-wave) may
# *not* be used here when using temporal wavefronts. The time loop may be
# found in StencilEquations::calc_rank().
RANK_LOOP_OPTS		?=	-dims 'dw,dx,dy,dz'
RANK_LOOP_OUTER_VARS	?=	dw,dx,dy,dz
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop($(RANK_LOOP_OUTER_VARS)) \
				{ $(RANK_LOOP_INNER_MODS) calc(region(start_dt, stop_dt, eqGroup_ptr)); }

# Region loops break up a region using OpenMP threading into blocks.  The
# 'omp' modifier creates an outer OpenMP loop so that each block is assigned
# to a top-level OpenMP thread.  The region time loops are not coded here to
# allow for proper spatial skewing for temporal wavefronts. The time loop
# may be found in StencilEquations::calc_region().
REGION_LOOP_OPTS	?=     	-dims 'rw,rx,ry,rz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_region_schedule)) proc_bind(spread)' \
				-calcPrefix 'eg->calc_'
REGION_LOOP_OUTER_VARS	?=	rw,rx,ry,rz
REGION_LOOP_OUTER_MODS	?=	grouped
REGION_LOOP_CODE	?=	omp $(REGION_LOOP_OUTER_MODS) loop($(REGION_LOOP_OUTER_VARS)) { \
				$(REGION_LOOP_INNER_MODS) calc(block(rt)); }

# Block loops break up a block into sub-blocks.  The 'omp' modifier creates
# a nested OpenMP loop so that each sub-block is assigned to a nested OpenMP
# thread.  There is no time loop here because threaded temporal blocking is
# not yet supported.
BLOCK_LOOP_OPTS		=     	-dims 'bw,bx,by,bz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)'
BLOCK_LOOP_OUTER_VARS	?=	bw,bx,by,bz
BLOCK_LOOP_OUTER_MODS	?=	grouped
BLOCK_LOOP_CODE		?=	omp $(BLOCK_LOOP_OUTER_MODS) loop($(BLOCK_LOOP_OUTER_VARS)) { \
				$(BLOCK_LOOP_INNER_MODS) calc(sub_block(bt)); }

# Sub-block loops break up a sub-block into vector clusters.  The indices at
# this level are by vector instead of element; this is indicated by the 'v'
# suffix. The innermost loop here is the final innermost loop. There is
# no time loop here because threaded temporal blocking is not yet supported.
SUB_BLOCK_LOOP_OPTS		=     	-dims 'sbwv,sbxv,sbyv,sbzv'
ifeq ($(split_L2),1)
 SUB_BLOCK_LOOP_OPTS		+=     	-splitL2
endif
SUB_BLOCK_LOOP_OUTER_VARS	?=	sbwv,sbxv,sbyv
SUB_BLOCK_LOOP_OUTER_MODS	?=	square_wave serpentine
SUB_BLOCK_LOOP_INNER_VARS	?=	sbzv
SUB_BLOCK_LOOP_INNER_MODS	?=	prefetch(L2)
SUB_BLOCK_LOOP_CODE		?=	$(SUB_BLOCK_LOOP_OUTER_MODS) loop($(SUB_BLOCK_LOOP_OUTER_VARS)) { \
					$(SUB_BLOCK_LOOP_INNER_MODS) loop($(SUB_BLOCK_LOOP_INNER_VARS)) { \
					calc(cluster(begin_sbtv)); } }

# Halo pack/unpack loops break up a region face, edge, or corner into vectors.
# The indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
# Nested OpenMP is not used here because there is no sharing between threads.
HALO_LOOP_OPTS		=     	-dims 'wv,xv,yv,zv' \
				-ompConstruct '$(omp_par_for) schedule($(omp_halo_schedule)) proc_bind(spread)'
HALO_LOOP_OUTER_MODS	?=	omp
HALO_LOOP_OUTER_VARS	?=	wv,xv,yv,zv
HALO_LOOP_CODE		?=	$(HALO_LOOP_OUTER_MODS) loop($(HALO_LOOP_OUTER_VARS)) \
				$(HALO_LOOP_INNER_MODS) { calc(halo(t)); }

#### Targets and rules ####

all:	$(EXEC_NAME) $(MAKE_REPORT_FILE)
	echo $(CXXFLAGS) > $(CXXFLAGS_FILE)
	echo $(LFLAGS) > $(LFLAGS_FILE)
	@cat $(MAKE_REPORT_FILE)
	@echo "Binary" $(EXEC_NAME) "has been built."
	@echo "Run command: bin/yask.sh -stencil" $(stencil) "-arch" $(arch) "[options]"

$(MAKE_REPORT_FILE): $(EXEC_NAME)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	$(MAKE) -j1 $(CODE_STATS) echo-settings >> $@ 2>&1

$(EXEC_NAME): $(STENCIL_OBJS)
	$(LD) -o $@ $(STENCIL_OBJS) $(CXXFLAGS) $(LFLAGS)

preprocess: $(STENCIL_CXX)

src/stencil_rank_loops.hpp: bin/gen-loops.pl Makefile
	$< -output $@ $(RANK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_RANK_LOOP_OPTS) "$(RANK_LOOP_CODE)"

src/stencil_region_loops.hpp: bin/gen-loops.pl Makefile
	$< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

src/stencil_block_loops.hpp: bin/gen-loops.pl Makefile
	$< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

src/stencil_sub_block_loops.hpp: bin/gen-loops.pl Makefile
	$< -output $@ $(SUB_BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_SUB_BLOCK_LOOP_OPTS) "$(SUB_BLOCK_LOOP_CODE)"

src/stencil_halo_loops.hpp: bin/gen-loops.pl Makefile
	$< -output $@ $(HALO_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_HALO_LOOP_OPTS) "$(HALO_LOOP_CODE)"

src/layout_macros.hpp: bin/gen-layouts.pl
	$< -m > $@

src/layouts.hpp: bin/gen-layouts.pl
	$< -d > $@

# Compile the stencil compiler.
# TODO: move this to its own makefile.
$(FB_EXEC): src/foldBuilder/*.*pp $(FB_STENCIL_LIST)
	$(FB_CXX) $(FB_CXXFLAGS) -o $@ src/foldBuilder/*.cpp $(EXTRA_FB_CXXFLAGS)

$(FB_STENCIL_LIST): src/foldBuilder/stencils/*.hpp
	echo '// Automatically-generated code; do not edit.' > $@
	for sfile in $(^F); do \
	  echo '#include "'$$sfile'"' >> $@; \
	done

# Run the stencil compiler and post-process its output.
$(ST_CODE_FILE): $(FB_EXEC)
	$< $(FB_FLAGS) $(EXTRA_FB_FLAGS) -p $(FB_TARGET) $@
	@- gindent -fca $@ || \
	  indent -fca $@ ||   \
	  echo "note:" $@ "not formatted."

$(ST_MACRO_FILE):
	echo '// Settings from YASK Makefile' > $@
	for macro in $(MACROS) $(EXTRA_MACROS); do \
	  echo '#define' $$macro | sed 's/=/ /' >> $@; \
	done

headers: $(GEN_HEADERS)
	@ echo 'Header files generated.'

%.$(TAG).o: %.cpp src/*.hpp src/foldBuilder/*.hpp $(GEN_HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.$(TAG).i: %.cpp src/*.hpp src/foldBuilder/*.hpp $(GEN_HEADERS)
	$(CXX) $(CXXFLAGS) -E $< > $@

# NB: All the following api* targets are temporary placeholders.
# TODO: Create better rules with real dependencies.

docs/latex/refman.tex: include/*hpp
	cd docs; doxygen yask_compiler.doxy

api-docs: docs/latex/refman.tex

src/foldBuilder/swig/yask_compiler.py: include/*hpp $(addprefix src/foldBuilder/,swig/setup.py *pp)
	cd src/foldBuilder/swig; \
	swig -v -I../../../include -c++ -python yask_compiler_api.i; \
	python setup.py build_ext --inplace

# TODO: build .so for compiler and link to it.
bin/yask_compiler_api_test.exe: $(addprefix src/foldBuilder/,tests/api_test.cpp *pp)
	$(FB_CXX) $(FB_CXXFLAGS) -o $@ \
	 $(addprefix src/foldBuilder/,tests/api_test.cpp [A-Z]*.cpp) $(EXTRA_FB_CXXFLAGS)

# Flags to avoid running stencil compiler.
API_MAKE_FLAGS := --new-file=$(FB_EXEC) --old-file=$(ST_CODE_FILE)
api-cxx-test: bin/yask_compiler_api_test.exe
	$(MAKE) clean
	$<
	mv api-cxx-test.dot src/foldBuilder/tests
	cd src/foldBuilder/tests; \
	dot -Tpdf -O api-cxx-test.dot; \
	ls -l api-cxx-test.dot*
	$(MAKE) -j $(API_MAKE_FLAGS) stencil=api-cxx-test arch=snb
	bin/yask.sh -stencil api-cxx-test -arch snb -v

api-py-test: src/foldBuilder/swig/yask_compiler.py
	$(MAKE) clean
	cd src/foldBuilder/tests; \
	python api_test.py; \
	dot -Tpdf -O api-py-test.dot; \
	ls -l api-py-test.dot*
	$(MAKE) -j $(API_MAKE_FLAGS) stencil=api-py-test arch=snb
	bin/yask.sh -stencil api-py-test -arch snb -v

api:	api-docs api-cxx-test api-py-test

tags:
	rm -f TAGS ; find . -name '*.[ch]pp' | xargs etags -C -a

clean:
	rm -fv src/*.[io] *.optrpt */*.optrpt *.s $(GEN_HEADERS) $(MAKE_REPORT_FILE)

realclean: clean
	rm -fv bin/*.exe make-report*.txt cxx-flags*.txt ld-flags.*txt $(FB_EXEC) TAGS $(FB_STENCIL_LIST)
	rm -fr docs/html docs/latex
	rm -fv src/foldBuilder/tests/*.dot*
	rm -fv stencil*.exe stencil-tuner-summary.csh stencil-tuner.pl gen-layouts.pl gen-loops.pl get-loop-stats.pl
	find . -name '*~' | xargs -r rm -v

echo-settings:
	@echo
	@echo "Build environment for" $(EXEC_NAME) on `date`
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
	@echo layout_wxyz=$(layout_wxyz)
	@echo layout_twxyz=$(layout_twxyz)
	@echo pfd_l1=$(pfd_l1)
	@echo pfd_l2=$(pfd_l2)
	@echo streaming_stores=$(streaming_stores)
	@echo omp_region_schedule=$(omp_region_schedule)
	@echo omp_block_schedule=$(omp_block_schedule)
	@echo omp_halo_schedule=$(omp_halo_schedule)
	@echo FB_TARGET="\"$(FB_TARGET)\""
	@echo FB_FLAGS="\"$(FB_FLAGS)\""
	@echo EXTRA_FB_FLAGS="\"$(EXTRA_FB_FLAGS)\""
	@echo MACROS="\"$(MACROS)\""
	@echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""
	@echo ISA=$(ISA)
	@echo OMPFLAGS="\"$(OMPFLAGS)\""
	@echo EXTRA_CXXFLAGS="\"$(EXTRA_CXXFLAGS)\""
	@echo CXX=$(CXX)
	@echo CXXOPT=$(CXXOPT)
	@echo CXXFLAGS="\"$(CXXFLAGS)\""
	@$(CXX) -v; $(CXX_VER_CMD)
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

code_stats:
	@echo
	@echo "Code stats for stencil computation:"
	bin/get-loop-stats.pl -t='sub_block_loops' *.s

help:
	@echo "Example usage:"
	@echo "make clean; make arch=knl stencil=iso3dfd"
	@echo "make clean; make arch=knl stencil=awp mpi=1"
	@echo "make clean; make arch=skx stencil=ave fold='x=1,y=2,z=4' cluster='x=2'"
	@echo "make clean; make arch=knc stencil=3axis radius=4 SUB_BLOCK_LOOP_INNER_MODS='prefetch(L1,L2)' pfd_l2=3"
	@echo " "
	@echo "Example debug usage:"
	@echo "make arch=knl  stencil=iso3dfd OMPFLAGS='-qopenmp-stubs' CXXOPT='-O0' EXTRA_MACROS='DEBUG'"
	@echo "make arch=intel64 stencil=ave OMPFLAGS='-qopenmp-stubs' CXXOPT='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo "make arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' OMPFLAGS='-qopenmp-stubs' EXTRA_MACROS='DEBUG DEBUG_TOLERANCE NO_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' CXXOPT='-O0'"
