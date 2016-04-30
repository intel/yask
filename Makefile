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

# stencil name: iso3dfd, 3axis, 9axis, 3plane, cube, ave, awp
stencil		= 	iso3dfd

# arch: target architecture: see options below.
arch		=	host

# defaults for various stencils:
#
# order:
# - historical term, not strictly order of PDE.
# - for most stencils, is width of spatial extent - 1.
# - for awp, only affects halo.
#
# real_bytes: FP precision: 4=float, 8=double.
#
# time_dim: allocated size of time dimension in grids.
#
# eqs: comma-separated name=substr pairs used to group
# grid update equations into stencil functions.
#
ifeq ($(stencil),ave)
order		=	2
real_bytes	=	8
else ifeq ($(stencil),9axis)
order		=	8
else ifeq ($(stencil),3plane)
order		=	6
else ifeq ($(stencil),cube)
order		=	4
else ifeq ($(stencil),awp)
order		=	4
time_dim	=	1
eqs		=	velocity=vel_,stress=stress_
endif

# general defaults for these vars if not set above.
order			?=	16
real_bytes		?=	4
time_dim		?=	2

# arch-specific settings:
#
# streaming_stores: Whether to use streaming stores.
#
# crew: whether to use Intel Crew threading feature.
# If you get linker errors about missing 'kmp*' symbols, your
# compiler does not support crew for the specified architecture.
# If you get a warning about 'num crews != num OpenMP threads',
# crew is not working properly. In either case, you should turn
# off crew by using 'crew=0' as a make argument.
#
# omp_schedule: OMP schedule policy.
ifeq ($(arch),)

$(error Architecture not specified; use arch=knc|knl|skx|hsw|ivb|snb|host)

else ifeq ($(arch),knc)

ISA		= 	-mmic
MACROS		+=	USE_INTRIN512
FB_TARGET  	=       knc
crew		=	1

else ifeq ($(arch),knl)

ISA		=	-xMIC-AVX512
MACROS		+=	USE_INTRIN512
FB_TARGET  	=       512
#crew		=	1  # enable for icc 2016 u3.

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

# general defaults for these vars if not set above.
crew			?= 	0
streaming_stores	?= 	1
omp_schedule		?=	dynamic
ISA			?=	-xHOST
FB_TARGET  		?=	cpp

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
FB_CC    	=       $(CC)
FB_CCFLAGS 	+=	-g -O1 -std=c++11 -Wall  # low opt to reduce compile time.
FB_FLAGS   	+=	-or $(order) -st $(stencil) -cluster $(cluster) -fold $(fold)
CC		=	icpc
FC		=	ifort
LD		=	$(CC)
MAKE		=	make
LFLAGS          +=    	-lpthread
CPPFLAGS        +=   	-g -O3 -std=c++11 -Wall $(ISA)
OMPFLAGS	+=	-fopenmp 
LFLAGS          +=       $(CPPFLAGS) -lrt -g
GEN_HEADERS     =	$(addprefix src/,stencil_outer_loops.hpp \
				stencil_region_loops.hpp stencil_block_loops.hpp \
				map_macros.hpp maps.hpp \
				stencil_macros.hpp stencil_code.hpp)
ifneq ($(eqs),)
  FB_FLAGS   	+=	-eq $(eqs)
endif

# settings passed as macros.
MACROS		+=	REAL_BYTES=$(real_bytes)
MACROS		+=	TIME_DIM=$(time_dim)

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
MACROS		+= 	ARCH_$(ARCH)

# compiler-specific settings
ifeq ($(notdir $(CC)),icpc)

CPPFLAGS        +=      -debug extended -Fa -restrict -ansi-alias -fno-alias
CPPFLAGS	+=	-fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -rcd -ftz -fma -fimf-domain-exclusion=none -qopt-assume-safe-padding
CPPFLAGS	+=      -qopt-report=5 -qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
CPPFLAGS	+=	-no-diag-message-catalog

# work around an optimization bug.
MACROS		+=	NO_STORE_INTRINSICS

ifeq ($(crew),1)
CPPFLAGS	+=      -mP2OPT_hpo_par_crew_codegen=T
MACROS		+=	__INTEL_CREW
endif

endif # compiler-specific

ifeq ($(streaming_stores),1)
MACROS		+=	USE_STREAMING_STORE
endif

# gen-loops.pl args for outer 3 sets of loops:

# Outer loops break up the whole problem into smaller regions.
OUTER_LOOP_OPTS		=	-dims 'dn,dx,dy,dz' -calcPrefix 'stencil->calc_'
OUTER_LOOP_CODE		=	loop(dn,dx,dy,dz) { calc(region); }

# Region loops break up a region using OpenMP threading into blocks.
REGION_LOOP_OPTS	=     	-dims 'rn,rx,ry,rz' -ompSchedule $(omp_schedule)
REGION_LOOP_CODE	=	serpentine omp loop(rn,rx,ry,rz) { calc(block); }

# Block loops break up a block into vector clusters.
# Note: the indices at this level are by vector instead of element;
# this is indicated by the 'v' suffix.
BLOCK_LOOP_OPTS		=     	-dims 'bnv,bxv,byv,bzv'
BLOCK_LOOP_CODE		=	loop(bnv) { crew loop(bxv) { loop(byv) { $(INNER_BLOCK_LOOP_OPTS) loop(bzv) { calc(cluster); } } } }


# compile with model_cache=1 or 2 to check prefetching.
ifeq ($(model_cache),1)
MACROS       	+=      MODEL_CACHE=1
OMPFLAGS	=	-qopenmp-stubs
else ifeq ($(model_cache),2)
MACROS       	+=      MODEL_CACHE=2
OMPFLAGS	=	-qopenmp-stubs
endif

CPPFLAGS	+=	$(addprefix -D,$(MACROS)) $(addprefix -D,$(EXTRA_MACROS)) $(OMPFLAGS) $(EXTRA_CPPFLAGS)
LFLAGS          +=      $(OMPFLAGS) $(EXTRA_CPPFLAGS)

STENCIL_BASES		:=	stencil_main stencil_calc utils
STENCIL_OBJS		:=	$(addprefix src/,$(addsuffix .$(arch).o,$(STENCIL_BASES)))
STENCIL_CPP		:=	$(addprefix src/,$(addsuffix .$(arch).i,$(STENCIL_BASES)))
STENCIL_EXEC_NAME	:=	stencil.$(arch).exe

ifeq ($(notdir $(CC)),icpc)
CODE_STATS      =   code_stats
endif

all:	$(STENCIL_EXEC_NAME) $(CODE_STATS) settings

settings:
	@echo
	@echo $(STENCIL_EXEC_NAME) "has been built."
	@echo
	@echo "Make settings used:"
	@echo arch=$(arch)
	@echo stencil=$(stencil)
	@echo fold=$(fold)
	@echo cluster=$(cluster)
	@echo order=$(order)
	@echo real_bytes=$(real_bytes)
	@echo time_dim=$(time_dim)
	@echo streaming_stores=$(streaming_stores)
	@echo FB_TARGET="\"$(FB_TARGET)\""
	@echo FB_FLAGS="\"$(FB_FLAGS)\""
	@echo EXTRA_FB_FLAGS="\"$(EXTRA_FB_FLAGS)\""
	@echo MACROS="\"$(MACROS)\""
	@echo EXTRA_MACROS="\"$(EXTRA_MACROS)\""
	@echo ISA=$(ISA)
	@echo OMPFLAGS="\"$(OMPFLAGS)\""
	@echo EXTRA_CPPFLAGS="\"$(EXTRA_CPPFLAGS)\""
	@echo CPPFLAGS="\"$(CPPFLAGS)\""
	@echo OUTER_LOOP_OPTS="\"$(OUTER_LOOP_OPTS)\""
	@echo OUTER_LOOP_CODE="\"$(OUTER_LOOP_CODE)\""
	@echo REGION_LOOP_OPTS="\"$(REGION_LOOP_OPTS)\""
	@echo REGION_LOOP_CODE="\"$(REGION_LOOP_CODE)\""
	@echo BLOCK_LOOP_OPTS="\"$(BLOCK_LOOP_OPTS)\""
	@echo BLOCK_LOOP_CODE="\"$(BLOCK_LOOP_CODE)\""

code_stats: $(STENCIL_EXEC_NAME)
	@echo
	@echo "Code stats for stencil computation:"
	./get-loop-stats.pl -t='block_loops' *.s
	@echo "Speedup estimates:"
	@grep speedup src/stencil_*.$(arch).optrpt | sort | uniq -c

$(STENCIL_EXEC_NAME): $(STENCIL_OBJS)
	$(LD) $(LFLAGS) -o $@ $(STENCIL_OBJS)

preprocess: $(STENCIL_CPP)

src/stencil_outer_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(OUTER_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_OUTER_LOOP_OPTS) "$(OUTER_LOOP_CODE)"

src/stencil_region_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(REGION_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_REGION_LOOP_OPTS) "$(REGION_LOOP_CODE)"

src/stencil_block_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(BLOCK_LOOP_OPTS) $(EXTRA_LOOP_OPTS) $(EXTRA_BLOCK_LOOP_OPTS) "$(BLOCK_LOOP_CODE)"

src/map_macros.hpp: gen-mapping.pl
	./$< -m > $@

src/maps.hpp: gen-mapping.pl
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
	rm -fv src/*.[io] *.optrpt src/*.optrpt *.s $(GEN_HEADERS) 

realclean: clean
	rm -fv stencil*.exe foldBuilder TAGS
	find . -name '*~' | xargs -r rm -v

help:
	@echo "Example usage:"
	@echo "make clean; make -j8 arch=knc"
	@echo "make clean; make -j8 arch=knl stencil=3axis order=8"
	@echo "make clean; make -j8 arch=skx stencil=ave fold='x=1,y=2,z=4' cluster='x=2' EXTRA_FB_FLAGS=-lus"
	@echo " "
	@echo "Example debug usage:"
	@echo "make arch=knl  OMPFLAGS='-qopenmp-stubs' crew=0 EXTRA_CPPFLAGS='-O0' EXTRA_MACROS='DEBUG'"
	@echo "make arch=host OMPFLAGS='-qopenmp-stubs' EXTRA_CPPFLAGS='-O0' EXTRA_MACROS='DEBUG' model_cache=2"
	@echo "make arch=host OMPFLAGS='-qopenmp-stubs' order=0 stencil=3axis fold='x=1,y=1,z=1' EXTRA_MACROS='DEBUG DEBUG_TOLERANCE EMU_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' EXTRA_CPPFLAGS='-O0'"
