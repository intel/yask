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
#   Currently, MPI is only used in X dimension.
#
# radius: sets size of certain stencils.
#
# real_bytes: FP precision: 4=float, 8=double.
#
# fold: How to fold vectors (x*y*z).
#   Vectorization in dimensions perpendicular to the inner loop
#   (defined by BLOCK_LOOP_CODE below) often works well.
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
# omp_schedule: OMP schedule policy for region loop.
# omp_block_schedule: OMP schedule policy for nested OpenMP block loop.
# omp_halo_schedule: OMP schedule policy for OpenMP halo loop.
#
# def_block_threads: Number of threads to use in nested OpenMP block loop by default.
# def_thread_divisor: Divide number of OpenMP threads by this factor by default.
#
# def_*_size, def_pad: Default sizes used in executable.

# Initial defaults.
stencil		=	unspecified
arch		=	knl
mpi		=	0

# Defaults based on stencil type.
ifeq ($(stencil),)
$(error Stencil not specified; use stencil=iso3dfd, 3axis, 9axis, 3plane, cube, ave, stream, awp, awp_elastic, ssg or fsg)

else ifeq ($(stencil),ave)
radius		?=	1
real_bytes	?=	8

else ifeq ($(stencil),3axis)
MACROS		+=	MAX_EXCH_DIST=1
radius		?=	6

else ifeq ($(stencil),9axis)
MACROS		+=	MAX_EXCH_DIST=2
radius		?=	4

else ifeq ($(stencil),3plane)
MACROS		+=	MAX_EXCH_DIST=2
radius		?=	3

else ifeq ($(stencil),cube)
MACROS		+=	MAX_EXCH_DIST=3
radius		?=	2

else ifeq ($(stencil),iso3dfd)
MACROS		+=	MAX_EXCH_DIST=1
radius		?=	8
real_bytes	?=	4
ifeq ($(arch),knl)
ifeq ($(real_bytes),4)
fold		?=	x=2,y=8,z=1
else
fold		?=	x=2,y=4,z=1
endif
cluster		?=	x=2
else
cluster		?=	z=2
endif

else ifeq ($(stencil),stream)
MACROS		+=	MAX_EXCH_DIST=0
radius		?=	2
cluster		?=	x=2

else ifneq ($(findstring awp,$(stencil)),)
eqs		?=	velocity=vel,stress=str
time_alloc	?=	1
def_block_size	?=	32
ifeq ($(arch),knl)
def_thread_divisor	?=	2
def_block_threads	?=	4
endif
FB_FLAGS	+=	-min-es 1

else ifeq ($(stencil)),ssg)
eqs		?=	v_bl=v_bl,v_tr=v_tr,v_tl=v_tl,s_br=s_br,s_bl=s_bl,s_tr=s_tr,s_tl=s_tl

else ifeq ($(stencil),fsg)
eqs             ?=      v_br=v_br,v_bl=v_bl,v_tr=v_tr,v_tl=v_tl,s_br=s_br,s_bl=s_bl,s_tr=s_tr,s_tl=s_tl
time_alloc	?=	1
fold            ?=      x=1,y=1,z=1
ifeq ($(arch),knl)
REGION_LOOP_CODE ?=     omp square_wave serpentine loop(rn,rz,rx,ry) { calc(block(rt)); }
BLOCK_LOOP_CODE ?=      omp square_wave serpentine loop(bnv,bxv,byv) { prefetch(L1) loop(bzv) { calc(cluster(bt)); } }
def_block_size  ?=      32
def_thread_divisor ?=	4
def_block_threads ?= 	1
def_pad		?=	2
endif

endif # stencil-specific.

# Defaut settings based on architecture.
ifeq ($(arch),knc)

ISA		?= 	-mmic
MACROS		+=	USE_INTRIN512
FB_TARGET  	?=       knc
def_block_threads ?=	4
BLOCK_LOOP_INNER_MODS	?=	prefetch(L1,L2)

else ifeq ($(arch),knl)

ISA		?=	-xMIC-AVX512
GCXX_ISA		?=	-march=knl
MACROS		+=	USE_INTRIN512 USE_RCP28
FB_TARGET  	?=       512
def_block_size	?=	96
def_block_threads ?=	8
streaming_stores  ?= 	0
BLOCK_LOOP_INNER_MODS	?=	prefetch(L1)

else ifeq ($(arch),skx)

ISA		?=	-xCORE-AVX512
GCXX_ISA		?=	-march=knl -mno-avx512er -mno-avx512pf
MACROS		+=	USE_INTRIN512
FB_TARGET  	?=       512

else ifeq ($(arch),hsw)

ISA		?=	-xCORE-AVX2
GCXX_ISA		?=	-march=haswell
MACROS		+=	USE_INTRIN256
FB_TARGET  	?=       256

else ifeq ($(arch),ivb)

ISA		?=	-xCORE-AVX-I
GCXX_ISA		?=	-march=ivybridge
MACROS		+=	USE_INTRIN256
FB_TARGET  	?=       256

else ifeq ($(arch),snb)

ISA		?=	-xAVX
GCXX_ISA		?=	-march=sandybridge
MACROS		+= 	USE_INTRIN256
FB_TARGET  	?=       256

else ifeq ($(arch),intel64)

ISA		?=	-xHOST
FB_TARGET	?=	cpp

else

$(error Architecture not recognized; use arch=knl, knc, skx, hsw, ivb, snb, or intel64 (no explicit vectorization))

endif # arch-specific.

# general defaults for vars if not set above.
streaming_stores	?= 	1
omp_par_for		?=	omp parallel for
omp_schedule		?=	dynamic,1
omp_block_schedule	?=	static,1
omp_halo_schedule	?=	static
def_block_threads	?=	2
def_thread_divisor	?=	1
real_bytes		?=	4
layout_xyz		?=	Layout_123
layout_txyz		?=	Layout_2314
layout_nxyz		?=	Layout_1234
layout_tnxyz		?=	Layout_23415
def_rank_size	?=	128
def_block_size		?=	64
def_pad			?=	1
cluster			?=	x=1

# default folding depends on HW vector size.
ifneq ($(findstring INTRIN512,$(MACROS)),)  # 512 bits.

ifeq ($(real_bytes),4)
fold		?=	x=4,y=4,z=1
else
fold		?=	x=4,y=2,z=1
endif

else  # not 512 bits.

ifeq ($(real_bytes),4)
fold		?=	x=8
else
fold		?=	x=4
endif

endif # not 512 bits.

# More build flags.
ifeq ($(mpi),1)
CXX		=	mpiicpc
else
CXX		=	icpc
endif
LD		=	$(CXX)
MAKE		=	make
CXXFLAGS        +=   	-g -O3 -std=c++11 -Wall
OMPFLAGS	+=	-fopenmp 
LFLAGS          +=      -lrt
FB_CXX    	=       $(CXX)
FB_CXXFLAGS 	+=	-g -O0 -std=c++11 -Wall  # low opt to reduce compile time.
EXTRA_FB_CXXFLAGS =
FB_FLAGS   	+=	-st $(stencil) -cluster $(cluster) -fold $(fold)
ST_MACRO_FILE	:=	stencil_macros.hpp
ST_CODE_FILE	:=	stencil_code.hpp
GEN_HEADERS     =	$(addprefix src/, \
				stencil_rank_loops.hpp \
				stencil_region_loops.hpp \
				stencil_halo_loops.hpp \
				stencil_block_loops.hpp \
				layout_macros.hpp layouts.hpp \
				$(ST_MACRO_FILE) $(ST_CODE_FILE) )
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


# Set more MACROS based on individual makefile vars.
# MACROS and EXTRA_MACROS will be written to a header file.
MACROS		+=	REAL_BYTES=$(real_bytes)
MACROS		+=	LAYOUT_XYZ=$(layout_xyz)
MACROS		+=	LAYOUT_TXYZ=$(layout_txyz)
MACROS		+=	LAYOUT_NXYZ=$(layout_nxyz)
MACROS		+=	LAYOUT_TNXYZ=$(layout_tnxyz)
MACROS		+=	DEF_RANK_SIZE=$(def_rank_size)
MACROS		+=	DEF_BLOCK_SIZE=$(def_block_size)
MACROS		+=	DEF_BLOCK_THREADS=$(def_block_threads)
MACROS		+=	DEF_THREAD_DIVISOR=$(def_thread_divisor)
MACROS		+=	DEF_PAD=$(def_pad)

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
CXXFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz -fma -fimf-domain-exclusion=none -qopt-assume-safe-padding -qoverride-limits -vec-threshold0
CXXFLAGS	+=      -qopt-report=5 -qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
CXXFLAGS	+=	-no-diag-message-catalog
CXX_VER_CMD	=	$(CXX) -V

# work around an optimization bug.
MACROS		+=	NO_STORE_INTRINSICS

else # not Intel compiler
CXXFLAGS	+=	$(GCXX_ISA) -Wno-unknown-pragmas -Wno-unused-variable

endif # compiler.

ifeq ($(streaming_stores),1)
MACROS		+=	USE_STREAMING_STORE
endif

# gen-loops.pl args:

# Rank loops break up the whole rank into smaller regions.
# In order for temporal wavefronts to operate properly, the
# order of spatial dimensions may be changed, but traversal
# paths that do not have strictly incrementing indices (e.g.,
# grouped, serpentine, square-wave) may not be used here when
# using temporal wavefronts. The time loop may be found
# in StencilEquations::calc_rank().
RANK_LOOP_OPTS		=	-dims 'dn,dx,dy,dz'
RANK_LOOP_CODE		?=	$(RANK_LOOP_OUTER_MODS) loop(dn,dx,dy,dz) \
				{ $(RANK_LOOP_INNER_MODS) calc(region(start_dt, stop_dt, eqGroup_ptr)); }

# Region loops break up a region using OpenMP threading into blocks.
# The region time loops are not coded here to allow for proper
# spatial skewing for temporal wavefronts. The time loop may be found
# in StencilEquations::calc_region().
REGION_LOOP_OPTS	=     	-dims 'rn,rx,ry,rz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_schedule)) proc_bind(spread)' \
				-calcPrefix 'eg->calc_'
REGION_LOOP_OUTER_MODS	?=	grouped omp
REGION_LOOP_CODE	?=	$(REGION_LOOP_OUTER_MODS) loop(rn,rx,ry,rz) \
				{ $(REGION_LOOP_INNER_MODS) calc(block(rt)); }

# Block loops break up a block into vector clusters.
# The indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
# The 'omp' modifier creates a nested OpenMP loop.
# There is no time loop here because threaded temporal blocking is not yet supported.
BLOCK_LOOP_OPTS		=     	-dims 'bnv,bxv,byv,bzv' \
				-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)'
BLOCK_LOOP_INNER_MODS	?=	prefetch(L2)
BLOCK_LOOP_OUTER_MODS	?=	omp
BLOCK_LOOP_CODE		?=	$(BLOCK_LOOP_OUTER_MODS) loop(bnv,bxv) { loop(byv) \
				{ $(BLOCK_LOOP_INNER_MODS) loop(bzv) { calc(cluster(bt)); } } }

# Halo pack/unpack loops break up a region face, edge, or corner into vectors.
# The indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
# Nested OpenMP is not used here because there is no sharing between threads.
HALO_LOOP_OPTS		=     	-dims 'nv,xv,yv,zv' \
				-ompConstruct '$(omp_par_for) schedule($(omp_halo_schedule)) proc_bind(spread)'
HALO_LOOP_OUTER_MODS	?=	omp
HALO_LOOP_CODE		?=	$(HALO_LOOP_OUTER_MODS) loop(nv,xv,yv,zv) \
				$(HALO_LOOP_INNER_MODS) { calc(halo(t)); }

# compile with model_cache=1 or 2 to check prefetching.
ifeq ($(model_cache),1)
MACROS       	+=      MODEL_CACHE=1
OMPFLAGS	=	-qopenmp-stubs
else ifeq ($(model_cache),2)
MACROS       	+=      MODEL_CACHE=2
OMPFLAGS	=	-qopenmp-stubs
endif

CXXFLAGS	+=	$(OMPFLAGS) $(EXTRA_CXXFLAGS)

STENCIL_BASES		:=	stencil_main stencil_calc realv_grids utils
STENCIL_OBJS		:=	$(addprefix src/,$(addsuffix .$(arch).o,$(STENCIL_BASES)))
STENCIL_CXX		:=	$(addprefix src/,$(addsuffix .$(arch).i,$(STENCIL_BASES)))
STENCIL_EXEC_NAME	:=	stencil.$(arch).exe
MAKE_REPORT_FILE	:=	make-report.txt
CXXFLAGS_FILE		:=	cxx-flags.txt
LFLAGS_FILE		:=	ld-flags.txt

all:	$(STENCIL_EXEC_NAME) $(MAKE_REPORT_FILE)
	echo $(CXXFLAGS) > $(CXXFLAGS_FILE)
	echo $(LFLAGS) > $(LFLAGS_FILE)
	@cat $(MAKE_REPORT_FILE)
	@echo $(STENCIL_EXEC_NAME) "has been built."

$(MAKE_REPORT_FILE): $(STENCIL_EXEC_NAME)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	$(MAKE) -j1 $(CODE_STATS) echo-settings >> $@ 2>&1

echo-settings:
	@echo
	@echo "Build environment for" $(STENCIL_EXEC_NAME) on `date`
	@echo arch=$(arch)
	@echo stencil=$(stencil)
	@echo fold=$(fold)
	@echo cluster=$(cluster)
	@echo radius=$(radius)
	@echo real_bytes=$(real_bytes)
	@echo layout_xyz=$(layout_xyz)
	@echo layout_txyz=$(layout_txyz)
	@echo layout_nxyz=$(layout_nxyz)
	@echo layout_tnxyz=$(layout_tnxyz)
	@echo streaming_stores=$(streaming_stores)
	@echo def_block_threads=$(def_block_threads)
	@echo omp_schedule=$(omp_schedule)
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
	@echo CXXFLAGS="\"$(CXXFLAGS)\""
	@echo RANK_LOOP_OPTS="\"$(RANK_LOOP_OPTS)\""
	@echo RANK_LOOP_OUTER_MODS="\"$(RANK_LOOP_OUTER_MODS)\""
	@echo RANK_LOOP_INNER_MODS="\"$(RANK_LOOP_INNER_MODS)\""
	@echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""
	@echo REGION_LOOP_OPTS="\"$(REGION_LOOP_OPTS)\""
	@echo REGION_LOOP_OUTER_MODS="\"$(REGION_LOOP_OUTER_MODS)\""
	@echo REGION_LOOP_INNER_MODS="\"$(REGION_LOOP_INNER_MODS)\""
	@echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""
	@echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""
	@echo BLOCK_LOOP_OUTER_MODS="\"$(BLOCK_LOOP_OUTER_MODS)\""
	@echo BLOCK_LOOP_INNER_MODS="\"$(BLOCK_LOOP_INNER_MODS)\""
	@echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""
	@echo HALO_LOOP_OPTS="\"$(HALO_LOOP_OPTS)\""
	@echo HALO_LOOP_OUTER_MODS="\"$(RANK_LOOP_OUTER_MODS)\""
	@echo HALO_LOOP_INNER_MODS="\"$(RANK_LOOP_INNER_MODS)\""
	@echo HALO_LOOP_CODE="\"$(HALO_LOOP_CODE)\""
	@echo CXX=$(CXX)
	@$(CXX) -v; $(CXX_VER_CMD)

code_stats:
	@echo
	@echo "Code stats for stencil computation:"
	./get-loop-stats.pl -t='block_loops' *.s

$(STENCIL_EXEC_NAME): $(STENCIL_OBJS)
	$(LD) -o $@ $(STENCIL_OBJS) $(CXXFLAGS) $(LFLAGS)

preprocess: $(STENCIL_CXX)

src/stencil_rank_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(RANK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_RANK_LOOP_OPTS) "$(RANK_LOOP_CODE)"

src/stencil_region_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

src/stencil_halo_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(HALO_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_HALO_LOOP_OPTS) "$(HALO_LOOP_CODE)"

src/stencil_block_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

src/layout_macros.hpp: gen-layouts.pl
	./$< -m > $@

src/layouts.hpp: gen-layouts.pl
	./$< -d > $@

# Compile the stencil compiler.
# TODO: move this to its own makefile.
foldBuilder: src/foldBuilder/*.*pp src/foldBuilder/stencils/*.*pp
	$(FB_CXX) $(FB_CXXFLAGS) -Isrc/foldBuilder/stencils -o $@ src/foldBuilder/*.cpp $(EXTRA_FB_CXXFLAGS)

# Run the stencil compiler and post-process its output files.
# Use the gmake pattern-rule trick to specify simultaneous targets.
%/$(ST_MACRO_FILE) %/$(ST_CODE_FILE): foldBuilder
	./$< $(FB_FLAGS) $(EXTRA_FB_FLAGS) \
	  -pm $*/$(ST_MACRO_FILE) -p$(FB_TARGET) $*/$(ST_CODE_FILE)
	echo >> $*/$(ST_MACRO_FILE)
	echo '// Settings from YASK Makefile' >> $*/$(ST_MACRO_FILE)
	for macro in $(MACROS) $(EXTRA_MACROS); do \
	  echo '#define' $$macro | sed 's/=/ /' >> $*/$(ST_MACRO_FILE); \
	done
	@- gindent -fca $*/$(ST_CODE_FILE) || \
	  indent -fca $*/$(ST_CODE_FILE) || \
	  echo "note:" $*/$(ST_CODE_FILE) "not formatted."

headers: $(GEN_HEADERS)
	@ echo 'Header files generated.'

%.$(arch).o: %.cpp src/*.hpp src/foldBuilder/*.hpp $(GEN_HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.$(arch).i: %.cpp src/*.hpp src/foldBuilder/*.hpp $(GEN_HEADERS)
	$(CXX) $(CXXFLAGS) -E $< > $@

tags:
	rm -f TAGS ; find . -name '*.[ch]pp' | xargs etags -C -a

clean:
	rm -fv src/*.[io] *.optrpt src/*.optrpt *.s $(GEN_HEADERS) $(MAKE_REPORT_FILE)

realclean: clean
	rm -fv stencil*.exe foldBuilder TAGS
	find . -name '*~' | xargs -r rm -v

help:
	@echo "Example usage:"
	@echo "make clean; make arch=knl stencil=iso3dfd"
	@echo "make clean; make arch=knl stencil=awp mpi=1"
	@echo "make clean; make arch=skx stencil=ave fold='x=1,y=2,z=4' cluster='x=2'"
	@echo "make clean; make arch=knc stencil=3axis radius=4 BLOCK_LOOP_INNER_MODS='prefetch(L1,L2)' EXTRA_MACROS='PFDL1=2 PFDL2=4'"
	@echo " "
	@echo "Example debug usage:"
	@echo "make arch=knl  stencil=iso3dfd OMPFLAGS='-qopenmp-stubs' EXTRA_CXXFLAGS='-O0' EXTRA_MACROS='DEBUG'"
	@echo "make arch=intel64 stencil=ave OMPFLAGS='-qopenmp-stubs' EXTRA_CXXFLAGS='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo "make arch=intel64 stencil=3axis radius=0 fold='x=1,y=1,z=1' OMPFLAGS='-qopenmp-stubs' EXTRA_MACROS='DEBUG DEBUG_TOLERANCE NO_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' EXTRA_CXXFLAGS='-O0'"
