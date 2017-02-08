#!/bin/bash

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

# Purpose: run stencil kernel in specified environment.

# Env vars to set.
envs="OMP_DISPLAY_ENV=VERBOSE OMP_PLACES=cores"
envs="$envs KMP_VERSION=1 KMP_HOT_TEAMS_MODE=1 KMP_HOT_TEAMS_MAX_LEVEL=2"

# Extra options for exe.
opts=""

unset arch
while true; do

    if [[ ! -n ${1+set} ]]; then
        break

    elif [[ "$1" == "-h" || "$1" == "-help" ]]; then
        opts="$opts -h"
        shift
        echo "$0 is a wrapper around the stencil executable to facilitate setting up the proper environment."
        echo "usage: $0 -arch <arch> [-mic <N>|-host <hostname>] [-sh_prefix <command>] [-exe_prefix <command>] [-ranks <N>] [<env-var=value>...] [[--] <executable options>]"
        echo " "
        if [[ -z ${arch:+ok} ]]; then
            echo "To see executable options, run '$0 -arch <arch> -- -help'."
        else
            echo "To see executable options, run '$0 -arch $arch -- -help'."
        fi
        echo " "
        echo "All options to be passed to the executable must be at the end of the command line."
        echo "The sh_prefix command is used to prefix a sub-shell."
        echo "The exe_prefix command is used to prefix the executable (set to 'true' to avoid actual run)."
        echo "If -host <hostname> is given, 'ssh <hostname>' will be pre-pended to the sh_prefix command."
        echo "If -ranks <N> is given, 'mpirun -n <N>' is pre-pended to the exe_prefix command,"
        echo " and -nrx <N> is passed to the executable;"
        echo " use -exe_prefix <command> explicitly if a different MPI command is needed,"
        echo " and/or override -nrx as needed."
        echo "If -arch 'knl' is given, it implies the following (which can be overridden):"
        echo " -exe_prefix 'numactl --preferred=1'"
        echo "If -mic <N> is given, it implies the following (which can be overridden):"
        echo " -arch 'knc'"
        echo " -host "`hostname`"-mic<N>"
        exit 1

    elif [[ "$1" == "-sh_prefix" && -n ${2+set} ]]; then
        sh_prefix=$2
        shift
        shift

    elif [[ "$1" == "-exe_prefix" && -n ${2+set} ]]; then
        exe_prefix=$2
        shift
        shift

    elif [[ "$1" == "-host" && -n ${2+set} ]]; then
        host=$2
        shift
        shift

    elif [[ "$1" == "-mic" && -n ${2+set} ]]; then
        arch="knc"
        host=`hostname`-mic$2
        shift
        shift

    elif [[ "$1" == "-arch" && -n ${2+set} ]]; then
        arch=$2
        shift
        shift

    elif [[ "$1" == "-ranks" && -n ${2+set} ]]; then
        nranks=$2
        opts="$opts -nrx $nranks"
        shift
        shift

    elif [[ "$1" =~ ^[A-Za-z0-9_]+= ]]; then
        envs="$envs $1"
        shift

    elif [[ "$1" == "--" ]]; then
        shift
        
        # will pass remaining options to executable.
        break

    else
        # will pass remaining options to executable.
        break
    fi

done                            # parsing options.

# check arch.
if [[ -z ${arch:+ok} ]]; then
    echo "error: must use -arch <arch>"
    exit 1
fi

# set defaults for KNL.
if [[ "$arch" == "knl" ]]; then
    true ${exe_prefix='numactl --preferred=1'}
fi

# MPI
if [[ -n "$nranks" ]]; then
    exe_prefix="mpirun -n $nranks $exe_prefix"
    envs="$envs I_MPI_PRINT_VERSION=1 I_MPI_DEBUG=5"
fi

# bail on errors past this point.
set -e

exe="stencil.$arch.exe"
if [[ ! -x $exe ]]; then
    echo "error: '$exe' not found or not executable."
    exit 1
fi

# additional settings w/special cases for KNC when no host specified.
if [[ $arch == "knc" && -z ${host+ok} ]]; then
    dir=/tmp/$USER
    icc=`which icc`
    iccdir=`dirname $icc`/../..
    libpath=":$iccdir/compiler/lib/mic"
    ssh $host "rm -rf $dir; mkdir -p $dir"
    scp $exe $host:$dir
else
    dir=`pwd`
    libpath=":$HOME/lib"
fi

# run on specified host
if [[ -n "$host" ]]; then
    sh_prefix="ssh $host $sh_prefix"
    envs="$envs PATH=$PATH LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH$libpath"

    nm=1
    while true; do
        echo "Verifying access to '$host'..."
        ping -c 1 $host && ssh $host uname -a && break
        echo "Waiting $nm min before trying again..."
        sleep $(( nm++ * 60 ))
    done
else
    envs="$envs LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH$libpath"
fi

# echo make report if it exists.
if [[ -e make-report.txt ]]; then
    cat make-report.txt
fi

# command sequence.
cmds="cd $dir; uname -a; lscpu; numactl -H; ldd ./$exe; env $envs $exe_prefix ./$exe $opts $*"

date
echo "==================="

if [[ -z "$sh_prefix" ]]; then
    sh -c -x "$cmds"
else
    echo "Running shell under '$sh_prefix'..."
    $sh_prefix "sh -c -x '$cmds'"
fi

date
