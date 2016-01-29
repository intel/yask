#!/bin/bash

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

# Purpose: run stencil kernel in specified environment.

# defaults.
arch=knl
affinity=balanced
exe_prefix=" "

if [ "$1" == "-h" -o "$1" == "-help" ]; then
  echo "usage: $0 [-sh_prefix <command>] [-exe_prefix <command>] [-mic <N>|-host <hostname>] [-arch <arch>] [-cores <N>] [-threads <N> (per core)] [-affinity <scatter|compact|balanced>] [executable options as shown below]"
  echo "Non-null defaults: arch='$arch', affinity='$affinity'."
  echo "Options to be passed to the executable must follow the ones above."
  echo "The sh_prefix command used to prefix the sub-shell."
  echo "The exe_prefix command is used to prefix the executable."
  echo "If -host <hostname> is given, 'ssh <hostname>' will be pre-pended to the sh_prefix command."
  echo "If -arch 'knl' is given, the default exe_prefix command will be 'numactl --preferred=1'."
  echo "If -mic <N> is given, the default arch will be 'knc', and the hostname will be set to the current hostname plus '-mic<N>."
fi

while true; do

if [ "$1" == "-sh_prefix" -a "$2" != "" ]; then
  sh_prefix=$2
  shift
  shift

elif [ "$1" == "-exe_prefix" -a "$2" != "" ]; then
  exe_prefix=$2
  shift
  shift

elif [ "$1" == "-host" -a "$2" != "" ]; then
  host=$2
  shift
  shift

elif [ "$1" == "-mic" -a "$2" != "" ]; then
  arch="knc"
  host=`hostname`-mic$2
  shift
  shift

elif [ "$1" == "-arch" -a "$2" != "" ]; then
  arch=$2
  shift
  shift

elif [ "$1" == "-cores" -a "$2" != "" ]; then
  ncores=$2
  shift
  shift

elif [ "$1" == "-threads" -a "$2" != "" ]; then
  nthreads=$2
  shift
  shift

elif [ "$1" == "-affinity" -a "$2" != "" ]; then
  affinity=$2
  shift
  shift

else
  break
fi

done                            # parsing options.

# set default exe_prefix for KNL.
if [ "$arch" == "knl" ] && [ "$exe_prefix" == " " ]; then
    exe_prefix='numactl --preferred=1'
fi


# set OMP thread vars if cores and/or threads were set.
if [[ -n "$ncores$nthreads" ]]; then

  if [ "$arch" == "knc" ]; then
    ncores=${ncores-61}
  else
    ncores=${ncores-68}
  fi
  nthreads=${nthreads-4}

  # set vars for both with and without crew.
  corestr="export OMP_NUM_THREADS=$(($ncores*$nthreads)); export KMP_PLACE_THREADS=${ncores}c,${nthreads}t; export INTEL_CREW_NUM_LEADERS=$ncores; export INTEL_CREW_SIZE=$nthreads;"
  echo "running with $ncores cores and $nthreads threads per core..."
else
  corestr=""
  echo "running with the default number of cores and threads..."
fi

set -e                          # bail on error.
icc=`which icc`
iccdir=`dirname $icc`/../..

exe="stencil.$arch.exe"
if [[ ! -x $exe ]]; then
    echo "error: '$exe' not found"
    exit 1
fi

if [[ ! -z $host ]]; then
    sh_prefix="ssh $host $sh_prefix"
    nm=1
    while true; do
        echo "Verifying access to '$host'..."
        ping -c 1 $host && ssh $host uname -a && break
        echo "Waiting $nm min before trying again..."
        sleep $(( nm++ * 60 ))
    done
fi

if [[ $arch == "knc" && "$host" != "" ]]; then
    dir=/tmp/$USER
    libpath=":$iccdir/compiler/lib/mic"
    ssh $host "rm -rf $dir; mkdir -p $dir"
    scp $exe $host:$dir
else
    dir=`pwd`
    libpath=":$HOME/lib"
    #libpath=":$HOME/lib:/opt/intel/knl/knl-a0-compiler_20150515/lib.f2i"
fi

# command sequence.
cmds="cd $dir; export KMP_VERSION=1; export PATH=$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH$libpath; $corestr export KMP_AFFINITY=$affinity; numactl -H; ldd ./$exe; $exe_prefix ./$exe $*"

date
echo "==================="

if [[ -z $sh_prefix ]]; then
    sh -c -x "$cmds"
else
    echo "Running shell under '$sh_prefix'..."
    $sh_prefix "sh -c -x '$cmds'"
fi

echo "==================="
date
