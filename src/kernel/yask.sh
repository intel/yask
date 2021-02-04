#!/bin/bash

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2020, Intel Corporation
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

# Create invocation string w/proper quoting.
invo="Script invocation: $0"
whitespace="[[:space:]]"
for i in "$@"
do
    if [[ $i =~ $whitespace ]]; then
        i=\'$i\'
    fi
    invo="$invo $i"
done

# Default env vars to print debug info.
envs="OMP_DISPLAY_ENV=VERBOSE"
envs+=" KMP_VERSION=1"
envs+=" I_MPI_PRINT_VERSION=1 I_MPI_DEBUG=5"

# On Cygwin, need to put lib dir in path to load .dll's.
if [[ `uname -o` == "Cygwin" ]]; then
	envs+=" PATH='$PATH':"`dirname $0`/../lib
fi

# Default arch.
cpu_flags=`grep -m1 '^flags' /proc/cpuinfo`
if [[ $cpu_flags =~ avx512dq ]]; then
    def_arch=avx512
elif [[ $cpu_flags =~ avx512pf ]]; then
    def_arch=knl
elif [[ $cpu_flags =~ avx2 ]]; then
    def_arch=avx2
elif [[ $cpu_flags =~ avx ]]; then
    def_arch=avx
else
    def_arch=intel64
fi
arch=$def_arch

# Default ranks.
nranks=1
if command -v numactl >/dev/null; then
    cpubind=`numactl -s | grep -m1 '^cpubind:'`
    if [[ -n "$cpubind" ]]; then
        cbwords=( $cpubind )
        nranks=$(( ${#cbwords[@]} - 1 ))
    fi
fi

# Other defaults.
pre_cmd=":"
post_cmd=""
helping=0
opts=""
bindir=`dirname $0`
logdir="./logs"
tmplog="/tmp/yask-p$$"

# Display stencils in this dir and exit.
function show_stencils {
    echo "Available stencil.arch combos in '$bindir' directory:"
    find $bindir -name 'yask_kernel.*.*.exe' | sed -e 's/.*yask_kernel\./ -stencil /' -e 's/\./ -arch /' -e 's/.exe//'
    echo "The default -arch argument for this host is '$def_arch'."
    exit 1
}

# Loop thru cmd-line args.
while true; do

    if [[ ! -n ${1+set} ]]; then
        break

    elif [[ "$1" == "-h" ]]; then
        shift
        echo "$0 is a wrapper around the YASK executable to set up the proper environment."
        echo "Usage: $0 -stencil <stencil> [options]"
        echo "  -stencil <stencil>"
        echo "     Specify the solution-name part of the kernel executable."
        echo "     Should correspond to stencil=<stencil> used during compilation."
        echo "     Run this script without any options to see the available stencils."
        echo " "
        echo "Some options are generic (parsed by the driver script and applied to any stencil),"
        echo " and some options are parsed by the stencil executable determined by stencil and arch."
        echo " "
        echo "Generic (script) options:"
        echo "  -h"
        echo "     Print this help."
        echo "     To see YASK stencil-specific options, run '$0 -stencil <stencil> [-arch <arch>] -help'."
        echo "  -arch <arch>"
        echo "     Specify the architecture-name part of the kernel executable."
        echo "     Overrides the default architecture determined from /proc/cpuinfo flags."
        echo "     The default <arch> for this host is '$def_arch'."
        echo "     Should correspond to arch=<arch> used during compilation."
        echo "  -host <hostname>"
        echo "     Specify host to run executable on."
        echo "     Run sub-shell under 'ssh <hostname>'."
        echo "  -sh_prefix <command>"
        echo "     Run sub-shell under <command>, e.g., a custom ssh command."
        echo "     If -host and -sh_prefix are both specified, run sub-shell under"
        echo "     'ssh <hostname> <command>."
        echo "  -exe <dir/file>"
        echo "     Specify <dir/file> as YASK executable instead of one in the same directory as"
        echo "     this script with a name based on stencil and arch."
        echo "     <dir>/../lib will also be prepended to the LD_LIBRARY_PATH env var."
        echo "  -exe_prefix <command>"
        echo "     Run YASK executable as an argument to <command>, e.g., 'numactl -N 0'."
        echo "  -pre_cmd <command(s)>"
        echo "     One or more commands to run before YASK executable."
        echo "  -post_cmd <command(s)>"
        echo "     One or more commands to run after YASK executable."
        echo "  -mpi_cmd <command>"
        echo "     Run <command> before the executable (and before the -exe_prefix argument)."
        echo "  -ranks <N>"
        echo "     Simplified MPI run (<N> ranks on current host)."
        echo "     Shortcut for the following options if <N> > 1:"
        echo "       -mpi_cmd 'mpirun -np <N>'"
        echo "     If a different MPI command is needed, use -mpi_cmd <command> explicitly."
        if [[ -n "$nranks" ]]; then
            echo "     The default <N> for this host is '$nranks'."
        fi
        echo "  -log <filename>"
        echo "     Write copy of output to <filename>."
        echo "     Default <filename> is based on stencil, arch, hostname, time-stamp, and process ID."
        echo "     Set to empty string ('') to avoid making a log."
        echo "  -log_dir <dir>"
        echo "     Directory name to prepend to log <filename>."
        echo "     Default is '$logdir'."
        echo "  -show_arch"
        echo "     Print the default architecture string and exit."
        echo "  <env-var=value>"
        echo "     Set environment variable <env-var> to <value>."
        echo "     Repeat as necessary to set multiple vars."
        exit 0

    elif [[ "$1" == "-help" ]]; then
        helping=1
        nranks=1
        logfile='/dev/null'

        # Pass option to executable.
        opts+=" $1"
        shift

    elif [[ "$1" == "-show_arch" ]]; then
        echo $def_arch
        exit 0

    elif [[ "$1" == "-stencil" && -n ${2+set} ]]; then
        stencil=$2
        shift
        shift

    elif [[ "$1" == "-arch" && -n ${2+set} ]]; then
        arch=$2
        shift
        shift

    elif [[ "$1" == "-host" && -n ${2+set} ]]; then
        host=$2
        shift
        shift

    elif [[ "$1" == "-sh_prefix" && -n ${2+set} ]]; then
        sh_prefix=$2
        shift
        shift

    elif [[ "$1" == "-mpi_cmd" && -n ${2+set} ]]; then
        mpi_cmd=$2
        shift
        shift

    elif [[ "$1" == "-pre_cmd" && -n ${2+set} ]]; then
        pre_cmd=$2
        shift
        shift

    elif [[ "$1" == "-post_cmd" && -n ${2+set} ]]; then
        post_cmd=$2
        shift
        shift

    elif [[ "$1" == "-exe" && -n ${2+set} ]]; then
        exe=$2
        bindir=`dirname $exe`
        shift
        shift

    elif [[ "$1" == "-exe_prefix" && -n ${2+set} ]]; then
        exe_prefix=$2
        shift
        shift

    elif [[ "$1" == "-log" && -n ${2+set} ]]; then
        logfile=$2
        if [[ -z "$logfile" ]]; then
            logfile=$tmplog
        fi
        shift
        shift

    elif [[ "$1" == "-log_dir" && -n ${2+set} ]]; then
        logdir=$2
        shift
        shift

    elif [[ "$1" == "-ranks" && -n ${2+set} ]]; then
        nranks=$2
        shift
        shift

    elif [[ "$1" =~ ^[A-Za-z0-9_]+= ]]; then
        envs+=" $1"
        shift

    elif [[ "$1" == "--" ]]; then
        shift

        # Pass all remaining options to executable and stop parsing.
        opts+=" $@"
        break

    else
        # Pass this unknown option to executable.
        opts+=" $1"
        shift
        
    fi

done                            # parsing options.
echo $invo

# Check required opt (yes, it's an oxymoron).
if [[ -z ${stencil:+ok} ]]; then
    echo "error: missing required option: -stencil <stencil>"
    show_stencils
fi

# Simplified MPI in x-dim only.
if [[ -n "$nranks" && $nranks > 1 ]]; then
    : ${mpi_cmd="mpirun -np $nranks"}
fi

# Bail on errors past this point, but only errors
# in this script, not in the executed commands.
set -e

# Actual host.
exe_host=${host:-`hostname`}

# Command to dump a file to stdout.
dump="head -v -n -0"

# Init log file.
: ${logfile:=yask.$stencil.$arch.$exe_host.`date +%Y-%m-%d_%H-%M`_p$$.log}
if [[ -n "$logdir" ]]; then
    logfile="$logdir/$logfile"
fi
echo "Writing log to '$logfile'."
mkdir -p `dirname $logfile`
echo $invo > $logfile

# These values must match the ones in Makefile.
# If the executable is built by overriding YK_TAG, YK_EXT_BASE, and/or
# YK_EXEC, this will fail.
tag=$stencil.$arch
: ${exe:="$bindir/yask_kernel.$tag.exe"}
make_report="$bindir/../build/yask_kernel.$tag.make-report.txt"
yc_report="$bindir/../build/yask_kernel.$tag.yask_compiler-report.txt"

# Double-check that exe exists.
if [[ ! -x $exe ]]; then
    echo "error: '$exe' not found or not executable." | tee -a $logfile
    show_stencils
fi

# Save most recent make report to log if it exists.
if [[ -e $make_report ]]; then
    $dump $make_report >> $logfile
    if  [[ -e $yc_report ]]; then
        $dump $yc_report >> $logfile
    fi
fi

dir=`pwd`
libpath=":$HOME/lib"

# Setup to run on specified host.
if [[ -n "$host" ]]; then
    sh_prefix="ssh $host $sh_prefix"
    envs+=" PATH=$PATH LD_LIBRARY_PATH=$bindir/../lib:$LD_LIBRARY_PATH$libpath"

    nm=1
    while true; do
        echo "Verifying access to '$host'..."
        ping -c 1 $host && ssh $host uname -a && break
        echo "Waiting $nm min before trying again..."
        sleep $(( nm++ * 60 ))
    done
else
    envs+=" LD_LIBRARY_PATH=$bindir/../lib:$LD_LIBRARY_PATH$libpath"
fi

# Thread reduction factor.
tdiv=2  # Assume 2-way HT on.
if [[ $arch == "knl" ]]; then
    tdiv=1
elif command -v lscpu >/dev/null; then
    nthr=`lscpu | awk '/Thread.s. per core:/ { print $4 }'`
    if [[ -n "$nthr" ]]; then
        tdiv=$nthr
    fi
fi
opts="-thread_divisor $tdiv $opts"

# Commands to capture some important system status and config info for benchmark documentation.
config_cmds="sleep 1; uptime; lscpu; cpuinfo -A; sed '/^$/q' /proc/cpuinfo; cpupower frequency-info; uname -a; $dump /etc/system-release; $dump /proc/cmdline; $dump /proc/meminfo; free -gt; numactl -H; ulimit -a"

# Command sequence to be run in a shell.
exe_str="$mpi_cmd $exe_prefix $exe $opts"
cmds="cd $dir; ulimit -s unlimited; $config_cmds; ldd $exe; date; $pre_cmd; env $envs $exe_str"
if [[ -n "$post_cmd" ]]; then
    cmds+="; $post_cmd"
fi
cmds+="; date"

echo "===================" | tee -a $logfile

# Finally, invoke the binary.
if [[ -z "$sh_prefix" ]]; then
    sh -c -x "$cmds" 2>&1 | tee -a $logfile
else
    echo "Running shell under '$sh_prefix'..."
    $sh_prefix "sh -c -x '$cmds'" 2>&1 | tee -a $logfile
fi
echo "===================" | tee -a $logfile

# Exit if just getting help.
if [[ $helping == 1 ]]; then
    exit 0
fi

function finish {
    if [[ "$logfile" == $tmplog ]]; then
        rm $tmplog
    else
        echo "Log saved in '$logfile'."
        echo "Run 'utils/bin/yask_log_to_csv.pl $logfile' to output in CSV format."
    fi
    exit $1
}

# Print invocation again.
echo $invo
exe_str="'$exe_str'"

# Return a non-zero exit condition if test failed.
if [[ `grep -c 'TEST FAILED' $logfile` > 0 ]]; then
    echo $exe_str did not pass internal validation test. | tee -a $logfile
    finish 1
fi

# Return a non-zero exit condition if executable didn't exit cleanly.
if [[ `grep -c 'YASK DONE' $logfile` == 0 ]]; then
    echo $exe_str did not exit cleanly. | tee -a $logfile
    finish 1
fi

# Print a message if test passed on at least one rank.
# (Script would have exited above if any rank failed.)
if [[ `grep -c 'TEST PASSED' $logfile` > 0 ]]; then
    echo $exe_str passed internal validation test. | tee -a $logfile
fi

# Print a final message, which will print if not validated or passed validation.
echo $exe_str ran successfully. | tee -a $logfile
finish 0

