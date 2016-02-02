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

# typical invocations:
# make -j8
# make -j8 arch=knl order=8
# make -j8 arch=skx shape=ave fold='2 1 4' cluster='2 1 1'

# example debug builds:
# make arch=knl  OMPFLAGS='-qopenmp-stubs'
# make arch=host OMPFLAGS='-qopenmp-stubs' model_cache=2 EXTRA_CPPFLAGS='-O1' MACROS='DEBUG'
# make arch=host OMPFLAGS='-qopenmp-stubs' order=0 shape=3axis fold='1 1 1' MACROS='DEBUG EMU_INTRINSICS TRACE TRACE_MEM TRACE_INTRINSICS' EXTRA_CPPFLAGS='-O0'

# stencil shape: iso3dfd, 3axis, 9axis, 3plane, cube, ave
shape		= 	iso3dfd

# number of time steps to calculate at once.
tsteps		=	1

# target architecture: knc, knl, skx, hsw, host.
# code is only functional (not optimized) for any arch w/o 512-bit vectors.
arch            = 	knc

# number of vars (grids) to solve.
vars		=	1

# stencil order.
order		=	16

# FP precision: 4=float, 8=double.
real_bytes	=	4

# defaults for miniGhost-like 27-point stencil.
ifeq ($(shape),ave)
vars		=	40
order		=	2
real_bytes	=	8
endif

# how to fold vectors (x*y*z)
ifeq ($(real_bytes),4)
fold		=	1 4 4
else
fold		=	1 2 4
endif
cluster		=	1 1 1

# heuristic for expression-size threshold.
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

# gen-loops args for outer 3 sets of loops:

# Outer loops break up the whole problem into smaller regions.
OUTER_LOOP_ARGS		=	-dims 'dv,dx,dy,dz' 
OUTER_LOOP_ARGS		+=	'loop(dv,dz,dy,dx) { calc(region); }'

# Region loops break up a region using OpenMP threading into blocks.
REGION_LOOP_ARGS	=     	-dims 'rv,rx,ry,rz' -ompSchedule $(omp_schedule)
REGION_LOOP_ARGS	+=	'serpentine omp loop(rv,rz,ry,rx) { calc(block); }'

# Block loops break up a block into vector clusters.
BLOCK_LOOP_ARGS		=     	-dims 'bv,bx,by,bz'
BLOCK_LOOP_ARGS		+=	'loop(bv) { crew loop(bz) { loop(by) { loop(bx) { calc(cluster); } } } }'

FB_CC    	=       $(CC)
FB_CCFLAGS 	=	-g -O0 -std=c++11 -Wall  # no opt to reduce compile time.
FB_FLAGS   	=	-ts $(tsteps) -or $(order) -st $(shape) -cluster $(cluster) -es $(expr_size)
CC		=	icpc
FC		=	ifort
LD		=	$(CC)
MAKE		=	make
LFLAGS          =    	-lpthread
CPPFLAGS        =   	-g -O3 -std=c++11 -Wall
OMPFLAGS	=	-fopenmp 
LFLAGS          =       $(CPPFLAGS) -lrt -g
GEN_HEADERS     =	$(addprefix src/,stencil_outer_loops.hpp stencil_region_loops.hpp stencil_block_loops.hpp mapping.hpp real_matrices.hpp stencil_code.hpp stencil_macros.hpp)

# real (FP) size.
CPPFLAGS	+=	-DREAL_BYTES=$(real_bytes)

# vars.
CPPFLAGS	+=	-DNUM_VARS=$(vars)

# arch.
ARCH		:=	$(shell echo $(arch) | tr '[:lower:]' '[:upper:]')
CPPFLAGS	+= 	-DARCH_$(ARCH)

# arch-specific settings
ifeq ($(arch),knc)

CPPFLAGS		+= -mmic
CPPFLAGS	+= 	-DINTRIN512
FB_FLAGS2  	=       -pknc $(fold)
crew		=	1

else ifeq ($(arch),knl)

CPPFLAGS	+=	-xMIC-AVX512
CPPFLAGS	+= 	-DINTRIN512
FB_FLAGS2  	=       -p512 $(fold)
crew		=	1

else ifeq ($(arch),skx)

CPPFLAGS	+=	-xCORE_AVX512
CPPFLAGS	+= 	-DINTRIN512
FB_FLAGS2  	=       -p512 $(fold)

else ifeq ($(arch),hsw)

CPPFLAGS	+=	-xCORE-AVX2
FB_FLAGS2   	=	-pcpp $(fold)

# any other arch.
else

CPPFLAGS	+=	-xHOST
FB_FLAGS2   	=	-pcpp $(fold)

endif # arch-specific.

# compiler-specific settings
ifeq ($(notdir $(CC)),icpc)

CPPFLAGS        +=      -debug -restrict -ansi-alias -fno-alias -fimf-precision=low -fast-transcendentals -no-prec-sqrt -no-prec-div -fp-model fast=2 -fno-protect-parens -opt-assume-safe-padding -Fa
CPPFLAGS	+=      -qopt-report=5 -qopt-report-phase=VEC,PAR,OPENMP,IPO,LOOP
CPPFLAGS	+=	-no-diag-message-catalog

ifeq ($(crew),1)
OPT_CREW	=	-D__INTEL_CREW -mP2OPT_hpo_par_crew_codegen=T
CPPFLAGS	+=      $(OPT_CREW)
endif

endif # compiler-specific

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

all:	$(STENCIL_EXEC_NAME) $(TAGS) $(CODE_STATS)
	@echo
	@echo $(STENCIL_EXEC_NAME) "has been built."
	@echo
	@echo "Make vars used:"
	@echo MACROS="'"$(MACROS)"'"
	@echo EXTRA_CPPFLAGS="'"$(EXTRA_CPPFLAGS)"'"
	@echo CPPFLAGS="'"$(CPPFLAGS)"'"

code_stats: $(STENCIL_EXEC_NAME)
	@echo
	@echo "Code stats for stencil computation:"
	@./get-loop-stats.pl stencil_calc.s
	@echo "Speedup estimates:"
	@grep speedup src/stencil_calc.$(arch).optrpt | sort | uniq -c

$(STENCIL_EXEC_NAME): $(STENCIL_OBJS)
	$(LD) $(LFLAGS) -o $@ $(STENCIL_OBJS)

src/stencil_outer_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(OUTER_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_OUTER_LOOP_ARGS) 

src/stencil_region_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(REGION_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_REGION_LOOP_ARGS) 

src/stencil_block_loops.hpp: gen-loops.pl Makefile
	./$< -output $@ $(BLOCK_LOOP_ARGS) $(EXTRA_LOOP_ARGS) $(EXTRA_BLOCK_LOOP_ARGS) 

src/mapping.hpp: gen-mapping.pl
	./$< -m > $@

src/real_matrices.hpp: gen-mapping.pl
	./$< -c > $@

foldBuilder: src/foldBuilder/*.cpp src/foldBuilder/*.hpp
	$(FB_CC) $(FB_CCFLAGS) -o $@ src/foldBuilder/*.cpp

src/stencil_macros.hpp: foldBuilder
	./$< $(FB_FLAGS) -pm $(fold) > $@

src/stencil_code.hpp: foldBuilder
	./$< $(FB_FLAGS) $(FB_FLAGS2) > $@
	gindent $@ || indent $@
	@grep -m1 -A3 Calculate $@

%.$(arch).o: %.cpp src/*.hpp src/foldBuilder/*.hpp $(GEN_HEADERS)
	$(CC) $(CPPFLAGS) -Isrc/foldBuilder -c -o $@ $<

TAGS: src/*.hpp src/*.cpp
	rm -f TAGS ; find . -name '*.[ch]pp' | xargs etags -C -a

clean:
	rm -fv src/*.o *.optrpt src/*.optrpt *.s $(GEN_HEADERS) 

realclean: clean
	rm -fv stencil*.exe foldBuilder TAGS
	find . -name '*~' | xargs -r rm -v
