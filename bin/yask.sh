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
echo "Invocation: $0 $@"

# Env vars to set.
envs="OMP_DISPLAY_ENV=VERBOSE OMP_PLACES=cores"
envs="$envs KMP_VERSION=1 KMP_HOT_TEAMS_MODE=1 KMP_HOT_TEAMS_MAX_LEVEL=2"
envs="$envs I_MPI_PRINT_VERSION=1 I_MPI_DEBUG=5"

# Extra options for exe.
opts=""

unset arch                      # Don't want to inherit from env.
while true; do

    if [[ ! -n ${1+set} ]]; then
        break

    elif [[ "$1" == "-h" || "$1" == "-help" ]]; then
        opts="$opts -h"
        shift
        echo "$0 is a wrapper around the stencil executable to facilitate setting up the proper environment."
        echo "usage: $0 -stencil <stencil> -arch <arch> [-mic <N>|-host <hostname>] [-sh_prefix <command>] [-exe_prefix <command>] [-ranks <N>] [<env-var=value>...] [[--] <executable options>]"
        echo " "
        if [[ -z ${stencil:+ok} || -z ${arch:+ok} ]]; then
            echo "To see executable options, run '$0 -stencil <stencil> -arch <arch> -- -help'."
        else
            echo "To see executable options, run '$0 -stencil $stencil -arch $arch -- -help'."
        fi
        echo " "
        echo "All options to be passed to the executable must be at the end of the command line."
        echo "The sh_prefix command is used to prefix a sub-shell."
        echo "The exe_prefix command is used to prefix the executable (set to 'true' to avoid actual run)."
        echo "If '-host <hostname>' is given, 'ssh <hostname>' will be pre-pended to the sh_prefix command."
        echo "The '-ranks' option is for simple one-socket x-dimension partitioning only."
        echo " If -ranks <N> is given, 'mpirun -n <N> -ppn <N>' is pre-pended to the exe_prefix command,"
        echo " and -nrx <N> is passed to the executable."
        echo " If a different MPI command or config is needed, use -exe_prefix <command> explicitly"
        echo " and -nr* options as needed."
        echo "If -arch 'knl' is given, it implies the following (which can be overridden):"
        echo " -exe_prefix 'numactl --preferred=1'"
        echo "If -mic <N> is given, it implies the following (which can be overridden):"
        echo " -arch 'knc'"
        echo " -host "`hostname`"-mic<N>"
        exit 1

    elif [[ "$1" == "-stencil" && -n ${2+set} ]]; then
        stencil=$2
        shift
        shift

    elif [[ "$1" == "-arch" && -n ${2+set} ]]; then
        arch=$2
        shift
        shift

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

# Check required opts.
if [[ -z ${stencil:+ok} ]]; then
    if [[ -z ${arch:+ok} ]]; then
        echo "error: missing required options: -stencil <stencil> -arch <arch>"
        exit 1
    fi
    echo "error: missing required option: -stencil <stencil>"
    exit 1
fi
if [[ -z ${arch:+ok} ]]; then
    echo "error: missing required option: -arch <arch>"
    exit 1
fi

# Set defaults for KNL.
# TODO: run numactl [on host] to determine if in flat mode.
if [[ "$arch" == "knl" ]]; then
    true ${exe_prefix='numactl --preferred=1'}
fi

# Simplified MPI in x-dim only.
if [[ -n "$nranks" ]]; then
    exe_prefix="mpirun -n $nranks -ppn $nranks $exe_prefix"
fi

# Bail on errors past this point.
set -e

# These values must match the ones in Makefile.
tag=$stencil.$arch
exe="bin/yask.$tag.exe"
make_report=make-report.$tag.txt

# Check for executable.
if [[ ! -x $exe ]]; then
    echo "'$exe' not found or not executable; trying to build with default settings..."
    make clean; make -j stencil=$stencil arch=$arch
fi
if [[ ! -x $exe ]]; then
    echo "error: '$exe' not found or not executable."
    exit 1
fi

# Additional settings w/special cases for KNC when no host specified.
if [[ $arch == "knc" && -z ${host+ok} ]]; then
    dir=/tmp/$USER
    icc=`which icc`
    iccdir=`dirname $icc`/../..
    libpath=":$iccdir/compiler/lib/mic"
    ssh $host "rm -rf $dir; mkdir -p $dir/bin"
    scp $exe $host:$dir/bin
else
    dir=`pwd`
    libpath=":$HOME/lib"
fi

# Run on specified host
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

# Print make report if it exists.
if [[ -e $make_report ]]; then
    echo "Build report:"
    cat $make_report
fi

# Command sequence.
cmds="cd $dir; uname -a; lscpu; numactl -H; ldd $exe; env $envs $exe_prefix $exe $opts $@"

date
echo "==================="

if [[ -z "$sh_prefix" ]]; then
    sh -c -x "$cmds"
else
    echo "Running shell under '$sh_prefix'..."
    $sh_prefix "sh -c -x '$cmds'"
fi

date
