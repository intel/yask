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

# Example usage:
# make -j8 arch=knc
# make -j8 arch=knl stencil=3axis order=8
# make -j8 arch=skx stencil=ave fold='x=1,y=2,z=4' cluster='x=2'

# Example debug usage:
# make arch=knl  OMPFLAGS='-qopenmp-stubs' EXTRA_CPPFLAGS='-O0' MACROS='DEBUG'
# make arch=host OMPFLAGS='-qopenmp-stubs' EXTRA_CPPFLAGS='-O0' MACROS='DEBUG' model_cache=2
# make arch=host OMPFLAGS='-qopenmp-stubs' order=0 stencil=3axis fold='x=1,y=1,z=1' MACROS='DEBUG DEBUG_TOLERANCE EMU_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' EXTRA_CPPFLAGS='-O0'

# target architecture: see options below.
arch            =	host

# stencil name: iso3dfd, 3axis, 9axis, 3plane, cube, ave, awp
stencil		= 	3axis

# stencil "order":
# - historical term, not strictly order of PDE.
# - for most stencils, is width of spatial extent - 1.
# - for awp, only affects halo.
order		=	16

# FP precision: 4=float, 8=double.
real_bytes	=	4

# allocated size of time dimension in grids.
time_dim	=	2

# default overrides for various stencils.
ifeq ($(stencil),ave)
order		=	2
real_bytes	=	8
endif
ifeq ($(stencil),9axis)
order		=	8
endif
ifeq ($(stencil),3plane)
order		=	6
endif
ifeq ($(stencil),cube)
order		=	4
endif
ifeq ($(stencil),awp)
order		=	4
time_dim	=	1
endif

# how to fold vectors (x*y*z)
ifeq ($(real_bytes),4)
fold		=	x=1,y=4,z=4
else
fold		=	x=1,y=2,z=4
endif

# how many vectors to compute at once.
cluster		=	x=1,y=1,z=1

# heuristic for expression-size threshold in foldBuilder.
expr_size	=	50

# OMP schedule policy.
omp_schedule	=	dynamic

# Whether to use Intel CREW.
# Off by default, but turned on by default below for
# certain architectures.
# If you get linker errors about missing 'kmp*' symbols, your
# compiler does not support crew for the specified architecture,
# and you should specify 'crew=0' as a make argument.
crew		= 	0

FB_CC    	=       $(CC)
FB_CCFLAGS 	=	-g -O0 -std=c++11 -Wall  # no opt to reduce compile time.
FB_FLAGS1   	=	-or $(order) -st $(stencil)
FB_FLAGS2   	=	-cluster $(cluster) -fold $(fold) -es $(expr_size)
CC		=	icpc
FC		=	ifort
LD		=	$(CC)
MAKE		=	make
LFLAGS          =    	-lpthread
CPPFLAGS        =   	-g -O3 -std=c++11 -Wall
OMPFLAGS	=	-fopenmp 
LFLAGS          =       $(CPPFLAGS) -lrt -g
GEN_HEADERS     =	$(addprefix src/,stencil_outer_loops.hpp \
				stencil_region_loops.hpp stencil_block_loops.hpp \
				map_macros.hpp maps.hpp \
				stencil_macros.hpp stencil_code.hpp)

# settings passed as macros.
CPPFLAGS	+=	-DREAL_BYTES=$(real_bytes)
CPPFLAGS	+=	-DTIME_DIM=$(time_dim)

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
CPPFLAGS	+= 	-DARCH_$(ARCH)

# arch-specific settings
ifeq ($(arch),)

$(error Architecture not specified; use arch=knc|knl|skx|hsw|bdw|ivb|snb|host)

else ifeq ($(arch),knc)

CPPFLAGS		+= -mmic
CPPFLAGS	+= 	-DINTRIN512
FB_TARGET  	=       knc
crew		=	1

else ifeq ($(arch),knl)

CPPFLAGS	+=	-xMIC-AVX512
CPPFLAGS	+= 	-DINTRIN512
FB_TARGET  	=       512
crew		=	1

else ifeq ($(arch),skx)

CPPFLAGS	+=	-xCORE_AVX512
CPPFLAGS	+= 	-DINTRIN512
FB_TARGET  	=       512

# any non-512-bit arch.
else

ifeq ($(arch),hsw)
CPPFLAGS	+=	-xCORE-AVX2
else ifeq ($(arch),bdw)
CPPFLAGS	+=	-xCORE-AVX2
else ifeq ($(arch),ivb)
CPPFLAGS	+=	-xCORE-AVX-I
else ifeq ($(arch),snb)
CPPFLAGS	+=	-xAVX
else
CPPFLAGS	+=	-xHOST
endif

FB_TARGET  	=       cpp

endif # arch-specific.

# compiler-specific settings
ifeq ($(notdir $(CC)),icpc)

CPPFLAGS        +=      -debug -restrict -ansi-alias -fno-alias -fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -qopt-assume-safe-padding -Fa
CPPFLAGS	+=      -qopt-report=5 -qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
CPPFLAGS	+=	-no-diag-message-catalog

ifeq ($(crew),1)
OPT_CREW	=	-D__INTEL_CREW -mP2OPT_hpo_par_crew_codegen=T
CPPFLAGS	+=      $(OPT_CREW)
endif

endif # compiler-specific

# gen-loops args for outer 3 sets of loops:

# Outer loops break up the whole problem into smaller regions.
OUTER_LOOP_ARGS		=	-dims 'dn,dx,dy,dz' -calcPrefix 'stencil->calc_'
OUTER_LOOP_ARGS		+=	'loop(dn,dz,dy,dx) { calc(region); }'

# Region loops break up a region using OpenMP threading into blocks.
REGION_LOOP_ARGS	=     	-dims 'rn,rx,ry,rz' -ompSchedule $(omp_schedule)
REGION_LOOP_ARGS	+=	'serpentine omp loop(rn,rz,ry,rx) { calc(block); }'

# Block loops break up a block into vector clusters.
# Note: the indices at this level are by vector instead of element.
BLOCK_LOOP_ARGS		=     	-dims 'bnv,bxv,byv,bzv'
BLOCK_LOOP_ARGS		+=	'loop(bnv) { crew loop(bzv) { loop(byv) {' $(CLUSTER_LOOP_OPTS) 'loop(bxv) { calc(cluster); } } } }'


# compile with model_cache=1 or 2 to check prefetching.
ifeq ($(model_cache),1)
CPPFLAGS       	+=      -DMODEL_CACHE=1
OMPFLAGS	=	-qopenmp-stubs
else ifeq ($(model_cache),2)
CPPFLAGS       	+=      -DMODEL_CACHE=2
OMPFLAGS	=	-qopenmp-stubs
endif

CPPFLAGS	+=	$(addprefix -D,$(MACROS)) $(OMPFLAGS) $(EXTRA_CPPFLAGS)
LFLAGS          +=      $(OMPFLAGS) $(EXTRA_CPPFLAGS)

STENCIL_OBJ_BASES	:=	stencil_main stencil_calc stencil_ref utils
STENCIL_OBJS		:=	$(addprefix src/,$(addsuffix .$(arch).o,$(STENCIL_OBJ_BASES)))
STENCIL_EXEC_NAME	:=	stencil.$(arch).exe

ifeq ($(notdir $(CC)),icpc)
CODE_STATS      =   code_stats
endif

all:	$(STENCIL_EXEC_NAME) $(CODE_STATS)
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
	@echo MACROS="'"$(MACROS)"'"
	@echo OMPFLAGS="'"$(OMPFLAGS)"'"
	@echo EXTRA_CPPFLAGS="'"$(EXTRA_CPPFLAGS)"'"
	@echo CPPFLAGS="'"$(CPPFLAGS)"'"

code_stats: $(STENCIL_EXEC_NAME)
	@echo
	@echo "Code stats for stencil computation:"
	./get-loop-stats.pl *.s
	@echo "Speedup estimates:"
	@grep speedup src/stencil_*.$(arch).optrpt | sort | uniq -c

$(STENCIL_EXEC_NAME): $(STENCIL_OBJS)
	$(LD) $(LFLAGS) -o $@ $(STENCIL_OBJS)

src/stencil_outer_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(OUTER_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_OUTER_LOOP_ARGS) 

src/stencil_region_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(REGION_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_REGION_LOOP_ARGS) 

src/stencil_block_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(BLOCK_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_BLOCK_LOOP_ARGS) 

src/map_macros.hpp: gen-mapping.pl
	./$< -m > $@

src/maps.hpp: gen-mapping.pl
	./$< -d > $@

foldBuilder: src/foldBuilder/*.*pp src/foldBuilder/stencils/*.*pp
	$(FB_CC) $(FB_CCFLAGS) -Isrc/foldBuilder/stencils -o $@ src/foldBuilder/*.cpp

src/stencil_macros.hpp: foldBuilder
	./$< $(FB_FLAGS1) $(FB_FLAGS2) -pm > $@

src/stencil_code.hpp: foldBuilder
	./$< $(FB_FLAGS1) $(FB_FLAGS2) -p$(FB_TARGET) > $@
	gindent $@ || indent $@

headers: $(GEN_HEADERS)
	@ echo 'Header files generated.'

%.$(arch).o: %.cpp src/*.hpp src/foldBuilder/*.hpp headers
	$(CC) $(CPPFLAGS) -c -o $@ $<

tags:
	rm -f TAGS ; find . -name '*.[ch]pp' | xargs etags -C -a

clean:
	rm -fv src/*.o *.optrpt src/*.optrpt *.s $(GEN_HEADERS) 

realclean: clean
	rm -fv stencil*.exe foldBuilder TAGS
	find . -name '*~' | xargs -r rm -v
