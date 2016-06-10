##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2016, Intel Corporation
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
# stencil: iso3dfd, 3axis, 9axis, 3plane, cube, ave, awp.
#
# arch: knc, knl, skx, hsw, ivb, snb, host (host = generic intel64).
#
# mpi: 0, 1: whether to use MPI. 
#   Currently, MPI is only used in X dimension.
#
# order:
# - historical term, not strictly order of PDE.
# - for most stencils, is width of spatial extent - 1.
#
# real_bytes: FP precision: 4=float, 8=double.
#
# time_dim_size: allocated size of time dimension in grids.
#
# eqs: comma-separated name=substr pairs used to group
#   grid update equations into stencil functions.
#
# streaming_stores: 0, 1: Whether to use streaming stores.
#
# crew: 0, 1: whether to use Intel Crew threading instead of nested OpenMP (deprecated).
#
# omp_schedule: OMP schedule policy for region loop.
#
# omp_block_schedule: OMP schedule policy for nested OpenMP block loop.
#
# def_block_threads: Number of threads to use in nested OpenMP block loop by default.
#
# def_*_size: Default sizes used in executable.

# Initial defaults.
stencil		=	unspecified
arch		=	host
mpi		=	0

# Defaults based on stencil type.
ifeq ($(stencil),)
$(error Stencil not specified; use stencil=iso3dfd, 3axis, 9axis, 3plane, cube, ave, or awp)

else ifeq ($(stencil),ave)
order		=	2
real_bytes	=	8
def_rank_size	=	256

else ifeq ($(stencil),9axis)
order		=	8

else ifeq ($(stencil),3plane)
order		=	6

else ifeq ($(stencil),cube)
order		=	4

else ifeq ($(stencil),awp)
order		=	4
time_dim_size	=	1
eqs		=	velocity=vel_,stress=stress_
def_rank_size	=	640
def_block_size	=	32
def_wavefront_region_size =	256
endif

# Defaut settings based on architecture.
ifeq ($(arch),)
$(error Architecture not specified; use arch=knl, knc, skx, hsw, ivb, snb, or host (host = generic intel64))

else ifeq ($(arch),knc)

ISA		= 	-mmic
MACROS		+=	USE_INTRIN512
FB_TARGET  	=       knc
def_block_threads =	4

else ifeq ($(arch),knl)

ISA		=	-xMIC-AVX512
MACROS		+=	USE_INTRIN512
FB_TARGET  	=       512
def_block_size	=	96
def_block_threads =	8

else ifeq ($(arch),skx)

ISA		=	-xCORE-AVX512
MACROS		+=	USE_INTRIN512
FB_TARGET  	=       512

else ifeq ($(arch),hsw)

ISA		=	-xCORE-AVX2
MACROS		+=	USE_INTRIN256
FB_TARGET  	=       256

else ifeq ($(arch),ivb)

ISA		=	-xCORE-AVX-I
MACROS		+=	USE_INTRIN256
FB_TARGET  	=       256

else ifeq ($(arch),snb)

ISA		=	-xAVX
MACROS		+= 	USE_INTRIN256
FB_TARGET  	=       256

endif # arch-specific.

# general defaults for vars if not set above.
crew				?= 	0
streaming_stores		?= 	1
omp_schedule			?=	dynamic
omp_block_schedule		?=	static,1
def_block_threads		?=	2
omp_par_for			?=	omp parallel for
ISA				?=	-xHOST
FB_TARGET  			?=	cpp
order				?=	16
real_bytes			?=	4
time_dim_size			?=	2
layout_3d			?=	Layout_123
layout_4d			?=	Layout_1234
def_rank_size			?=	1024
def_block_size			?=	32
def_wavefront_region_size	?=	512

# How to fold vectors (x*y*z).
# Vectorization in dimensions perpendicular to the inner loop
# (defined by BLOCK_LOOP_CODE below) often works well.

ifneq ($(findstring INTRIN512,$(MACROS)),)  # 512 bits.
ifeq ($(real_bytes),4)
fold		=	x=4,y=4,z=1
else
fold		=	x=4,y=2,z=1
endif

else  # not 512 bits.
ifeq ($(real_bytes),4)
fold		=	x=8
else
fold		=	x=4
endif

endif

# How many vectors to compute at once (unrolling factor in
# each dimension).
cluster		=	x=1,y=1,z=1

# More build flags.
ifeq ($(mpi),1)
CC		=	mpiicpc
else
CC		=	icpc
endif
FC		=	ifort
LD		=	$(CC)
MAKE		=	make
LFLAGS          +=    	-lpthread
CPPFLAGS        +=   	-g -O3 -std=c++11 -Wall $(ISA)
OMPFLAGS	+=	-fopenmp 
LFLAGS          +=       $(CPPFLAGS) -lrt -g
FB_CC    	=       icpc
FB_CCFLAGS 	+=	-g -O1 -std=c++11 -Wall  # low opt to reduce compile time.
FB_FLAGS   	+=	-or $(order) -st $(stencil) -cluster $(cluster) -fold $(fold)
GEN_HEADERS     =	$(addprefix src/, \
				stencil_macros.hpp stencil_code.hpp \
				stencil_rank_loops.hpp \
				stencil_region_loops.hpp \
				stencil_halo_loops.hpp \
				stencil_block_loops.hpp \
				layout_macros.hpp layouts.hpp )
ifneq ($(eqs),)
  FB_FLAGS   	+=	-eq $(eqs)
endif

# set macros based on vars.
MACROS		+=	REAL_BYTES=$(real_bytes)
MACROS		+=	LAYOUT_3D=$(layout_3d)
MACROS		+=	LAYOUT_4D=$(layout_4d)
MACROS		+=	TIME_DIM_SIZE=$(time_dim_size)
MACROS		+=	DEF_RANK_SIZE=$(def_rank_size)
MACROS		+=	DEF_BLOCK_SIZE=$(def_block_size)
MACROS		+=	DEF_WAVEFRONT_REGION_SIZE=$(def_wavefront_region_size)
MACROS		+=	DEF_BLOCK_THREADS=$(def_block_threads)

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
MACROS		+= 	ARCH_$(ARCH)

# MPI settings.
ifeq ($(mpi),1)
MACROS		+=	USE_MPI
crew		=	0
endif

# compiler-specific settings
ifneq ($(findstring ic,$(notdir $(CC))),)

CODE_STATS      =   	code_stats
CPPFLAGS        +=      -debug extended -Fa -restrict -ansi-alias -fno-alias
CPPFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz -fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
CPPFLAGS	+=      -qopt-report=5 -qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
CPPFLAGS	+=	-no-diag-message-catalog
CC_VER_CMD	=	$(CC) -V

# work around an optimization bug.
MACROS		+=	NO_STORE_INTRINSICS

ifeq ($(crew),1)
CPPFLAGS	+=      -mP2OPT_hpo_par_crew_codegen=T
MACROS		+=	__INTEL_CREW
def_block_threads =	1
endif

endif # icpc

ifeq ($(streaming_stores),1)
MACROS		+=	USE_STREAMING_STORE
endif

# gen-loops.pl args for outer 3 sets of loops:

# Rank loops break up the whole rank into smaller regions.
# In order for tempral wavefronts to operate properly, the
# order of spatial dimensions may be changed, but traversal
# paths that do not have strictly incrementing indices (such as
# serpentine and/or square-wave) may not be used here when
# using temporal wavefronts.
RANK_LOOP_OPTS		=	-dims 'dt,dn,dx,dy,dz'
RANK_LOOP_CODE		=	loop(dt) { loop(dn,dx,dy,dz) { calc(region); } }

# Region loops break up a region using OpenMP threading into blocks.
# The region time loops are not coded here to allow for proper
# spatial skewing for temporal wavefronts. The time loop may be found
# in StencilEquations::calc_region().
REGION_LOOP_OPTS	=     	-dims 'rn,rx,ry,rz' \
				-ompConstruct '$(omp_par_for) schedule($(omp_schedule)) proc_bind(spread)' \
				-calcPrefix 'stencil->calc_'
REGION_LOOP_CODE	=	serpentine omp loop(rn,rx,ry,rz) { calc(block(rt)); }

# Block loops break up a block into vector clusters.
# The indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
# There is no time loop here because temporal blocking is not yet supported.
# The 'omp' modifier creates a nested OpenMP loop.
BLOCK_LOOP_OPTS		=     	-dims 'bnv,bxv,byv,bzv'
BLOCK_LOOP_INNER_MODS	=	prefetch(L1)
ifeq ($(crew),1)
BLOCK_LOOP_OUTER_MODS	=	crew
else
BLOCK_LOOP_OUTER_MODS	=	omp
BLOCK_LOOP_OPTS		+=	-ompConstruct '$(omp_par_for) schedule($(omp_block_schedule)) proc_bind(close)'
endif
BLOCK_LOOP_CODE		=	$(BLOCK_LOOP_OUTER_MODS) loop(bnv,bxv) { loop(byv) { \
				$(BLOCK_LOOP_INNER_MODS) loop(bzv) { calc(cluster(bt)); } } }

# Halo pack/unpack loops break up a region slice into vectors.
# The indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
# Note that there are no nested OpenMP loops here.
HALO_LOOP_OPTS		=     	-dims 'rnv,rxv,ryv,rzv' \
				-ompConstruct '$(omp_par_for) schedule($(omp_schedule))'
HALO_LOOP_CODE		=	serpentine omp loop(rnv,rxv,ryv) { loop(rzv) { calc(halo(rt)); } }

# compile with model_cache=1 or 2 to check prefetching.
ifeq ($(model_cache),1)
MACROS       	+=      MODEL_CACHE=1
OMPFLAGS	=	-qopenmp-stubs
else ifeq ($(model_cache),2)
MACROS       	+=      MODEL_CACHE=2
OMPFLAGS	=	-qopenmp-stubs
endif

CPPFLAGS	+=	$(addprefix -D,$(MACROS)) $(addprefix -D,$(EXTRA_MACROS))
CPPFLAGS	+=	$(OMPFLAGS) $(EXTRA_CPPFLAGS)
LFLAGS          +=      $(OMPFLAGS) $(EXTRA_CPPFLAGS)

STENCIL_BASES		:=	stencil_main stencil_calc utils
STENCIL_OBJS		:=	$(addprefix src/,$(addsuffix .$(arch).o,$(STENCIL_BASES)))
STENCIL_CPP		:=	$(addprefix src/,$(addsuffix .$(arch).i,$(STENCIL_BASES)))
STENCIL_EXEC_NAME	:=	stencil.$(arch).exe
MAKE_VAR_FILE		:=	make-vars.txt

all:	$(STENCIL_EXEC_NAME) $(MAKE_VAR_FILE)
	@cat $(MAKE_VAR_FILE)
	@echo $(STENCIL_EXEC_NAME) "has been built."

$(MAKE_VAR_FILE): $(STENCIL_EXEC_NAME)
	@echo MAKEFLAGS="\"$(MAKEFLAGS)"\" > $@ 2>&1
	$(MAKE) -j1 $(CODE_STATS) echo-settings >> $@ 2>&1

echo-settings:
	@echo
	@echo "Build environment for" $(STENCIL_EXEC_NAME) on `date`
	@echo arch=$(arch)
	@echo stencil=$(stencil)
	@echo fold=$(fold)
	@echo cluster=$(cluster)
	@echo order=$(order)
	@echo real_bytes=$(real_bytes)
	@echo layout_3d=$(layout_3d)
	@echo layout_4d=$(layout_4d)
	@echo time_dim_size=$(time_dim_size)
	@echo streaming_stores=$(streaming_stores)
	@echo omp_schedule=$(omp_schedule)
	@echo def_block_threads=$(def_block_threads)
	@echo omp_block_schedule=$(omp_block_schedule)
	@echo FB_TARGET="\"$(FB_TARGET)\""
	@echo FB_FLAGS="\"$(FB_FLAGS)\""
	@echo EXTRA_FB_FLAGS="\"$(EXTRA_FB_FLAGS)\""
	@echo MACROS="\"$(MACROS)\""
	@echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""
	@echo ISA=$(ISA)
	@echo OMPFLAGS="\"$(OMPFLAGS)\""
	@echo EXTRA_CPPFLAGS="\"$(EXTRA_CPPFLAGS)\""
	@echo CPPFLAGS="\"$(CPPFLAGS)\""
	@echo RANK_LOOP_OPTS="\"$(RANK_LOOP_OPTS)\""
	@echo RANK_LOOP_CODE="\"$(RANK_LOOP_CODE)\""
	@echo REGION_LOOP_OPTS="\"$(REGION_LOOP_OPTS)\""
	@echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""
	@echo INNER_BLOCK_LOOP_OPTS="\"$(INNER_BLOCK_LOOP_OPTS)\""
	@echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""
	@echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""
	@echo HALO_LOOP_OPTS="\"$(HALO_LOOP_OPTS)\""
	@echo HALO_LOOP_CODE="\"$(HALO_LOOP_CODE)\""
	@$(CC) -v; $(CC_VER_CMD)

code_stats:
	@echo
	@echo "Code stats for stencil computation:"
	./get-loop-stats.pl -t='block_loops' *.s

$(STENCIL_EXEC_NAME): $(STENCIL_OBJS)
	$(LD) $(LFLAGS) -o $@ $(STENCIL_OBJS)

preprocess: $(STENCIL_CPP)

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

foldBuilder: src/foldBuilder/*.*pp src/foldBuilder/stencils/*.*pp
	$(FB_CC) $(FB_CCFLAGS) -Isrc/foldBuilder/stencils -o $@ src/foldBuilder/*.cpp

src/stencil_macros.hpp: foldBuilder
	./$< $(FB_FLAGS) $(EXTRA_FB_FLAGS) -pm > $@

src/stencil_code.hpp: foldBuilder
	./$< $(FB_FLAGS) $(EXTRA_FB_FLAGS) -p$(FB_TARGET) > $@
	- gindent $@ || indent $@ || echo "note: no indent program found"

headers: $(GEN_HEADERS)
	@ echo 'Header files generated.'

%.$(arch).o: %.cpp src/*.hpp src/foldBuilder/*.hpp headers
	$(CC) $(CPPFLAGS) -c -o $@ $<

%.$(arch).i: %.cpp src/*.hpp src/foldBuilder/*.hpp headers
	$(CC) $(CPPFLAGS) -E $< > $@

tags:
	rm -f TAGS ; find . -name '*.[ch]pp' | xargs etags -C -a

clean:
	rm -fv src/*.[io] *.optrpt src/*.optrpt *.s $(GEN_HEADERS) $(MAKE_VAR_FILE)

realclean: clean
	rm -fv stencil*.exe foldBuilder TAGS
	find . -name '*~' | xargs -r rm -v

help:
	@echo "Example usage:"
	@echo "make clean; make arch=knc stencil=iso3dfd"
	@echo "make clean; make arch=knl stencil=3axis order=8 INNER_BLOCK_LOOP_OPTS='prefetch(L1,L2)'"
	@echo "make clean; make arch=skx stencil=ave fold='x=1,y=2,z=4' cluster='x=2'"
	@echo " "
	@echo "Example debug usage:"
	@echo "make arch=knl  stencil=iso3dfd OMPFLAGS='-qopenmp-stubs' EXTRA_CPPFLAGS='-O0' EXTRA_MACROS='DEBUG'"
	@echo "make arch=host stencil=ave OMPFLAGS='-qopenmp-stubs' EXTRA_CPPFLAGS='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo "make arch=host stencil=3axis order=0 fold='x=1,y=1,z=1' OMPFLAGS='-qopenmp-stubs' EXTRA_MACROS='DEBUG DEBUG_TOLERANCE NO_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' EXTRA_CPPFLAGS='-O0'"
