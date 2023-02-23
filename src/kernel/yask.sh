#!/bin/bash

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2023, Intel Corporation
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
for i in "$@"; do
    if [[ $i =~ [[:space:]] ]]; then
        i=\'$i\'
    fi
    invo="$invo $i"
done

# Default env vars to print debug info and set CPU and mem-binding.
# https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/environment-variables-for-memory-policy-control.html
envs="OMP_DISPLAY_ENV=VERBOSE KMP_VERSION=1"
envs+=" OMP_PLACES=cores KMP_HOT_TEAMS_MODE=1 KMP_HOT_TEAMS_MAX_LEVEL=3"
envs+=" I_MPI_PRINT_VERSION=1 I_MPI_DEBUG=5"
envs+=" I_MPI_HBW_POLICY=hbw_preferred,hbw_preferred"

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

# Default nodes.
nnodes=1
if [[ ! -z ${SLURM_JOB_NUM_NODES:+x} ]]; then
    nnodes=$SLURM_JOB_NUM_NODES
fi

# Default MPI ranks.
# Try Slurm var, then numactl, then lscpu.
# For latter two, the goal is to count only NUMA nodes with CPUs.
# (Systems with HBM may have NUMA nodes without CPUs.)
nranks=1
if [[ ! -z ${SLURM_NTASKS:+x} ]]; then
    nranks=$SLURM_NTASKS
elif command -v numactl >/dev/null; then
    ncpubinds=`numactl -s | awk '/^cpubind:/ { print NF-1 }'`
    if [[ -n "$ncpubinds" ]]; then
        nranks=$ncpubinds
    fi
elif command -v lscpu >/dev/null; then
    nnumas=`lscpu | grep -c '^NUMA node.*CPU'`
    if [[ -n "$nnumas" ]]; then
        nranks=$nnumas
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

# Validation shortcut (-v) vars.
doval=0
val="-validate -no-pre_auto_tune -no-auto_tune -no-warmup -num_trials 1 -trial_steps 2 -l 80 -Mb 72 -b 64 -mb 56 -nb 48 -pb 20"

# Display stencils in this dir and exit.
function show_stencils {
    echo "Available stencil.arch combos in '$bindir' directory:"
    find $bindir -name 'yask_kernel.*.*.exe' | sed -e 's/.*yask_kernel\./ -stencil /' -e 's/\./ -arch /' -e 's/.exe//'
    echo "The default -arch argument for this host is '$def_arch'."
    exit 1
}

# Loop thru cmd-line args.
using_opt_outer_threads=0
while true; do

    if [[ ! -n ${1+set} ]]; then
        break

    elif [[ "$1" == "-h" ]]; then
        shift
        echo "$0 is a wrapper around the YASK executable to set up the proper environment."
        echo "Usage: $0 -stencil <name> [options]"
        echo "  -stencil <name>"
        echo "     Specify the solution-name part of the YASK executable."
        echo "     Should correspond to stencil=<name> used during compilation,"
        echo "     or YK_STENCIL=<name> if that was used to override the default."
        echo "     Run this script without any options to see the available stencils."
        echo " "
        echo "Script options:"
        echo "  -h"
        echo "     Print this help."
        echo "     To see options from the YASK kernel executable, run the following command:"
        echo "       $0 -stencil <name> [-arch <name>] -help"
        echo "     This will run the YASK executable with the '-help' option."
        echo "  -arch <name>"
        echo "     Specify the architecture-name part of the YASK executable."
        echo "     Overrides the default architecture determined from /proc/cpuinfo flags."
        echo "     The default arch for this host is '$def_arch'."
        echo "     Should correspond to arch=<name> used during compilation"
        echo "     with '.offload-<offload_arch>' appended when built with 'offload=1',"
        echo "     or YK_ARCH=<name> if that was used to override the default."
        echo "     In any case, the '-stencil' and '-arch' args required to launch"
        echo "     any executable are printed at the end of a successful compilation."
        echo "  -host <hostname>"
        echo "     Specify host to run YASK executable on."
        echo "     Run sub-shell under 'ssh <hostname>'."
        echo "  -sh_prefix <command>"
        echo "     Run sub-shell under <command>, e.g., a custom ssh command."
        echo "     If -host and -sh_prefix are both specified, run sub-shell under"
        echo "     'ssh <hostname> <command>'."
        echo "  -exe <dir/file>"
        echo "     Specify <dir/file> as YASK executable instead of one in the same directory as"
        echo "     this script with a name based on '-stencil' and '-arch'."
        echo "     <dir>/../lib will also be prepended to the LD_LIBRARY_PATH env var."
        echo "  -exe_prefix <command>"
        echo "     Run YASK executable as an argument to <command>, e.g., 'numactl -N 0'."
        echo "  -mpi_cmd <command>"
        echo "     Run YASK executable as an argument to <command>, e.g., 'mpiexec.hydra -n 4'."
        echo "     If -mpi_cmd is used, the -ranks option is used only for computing the"
        echo "       default number of OpenMP threads to use."
        echo "     If -mpi_cmd and -exe_prefix are both specified, this one is used first."
        echo "  -ranks <N>"
        echo "     Run the YASK executable on <N> MPI ranks."
        echo "     Shortcut for the following option if <N> > 1:"
        echo "       -mpi_cmd 'mpirun -np <N>'"
        echo "     If a different MPI command is needed, use -mpi_cmd <command> explicitly."
        echo "     If the env var SLURM_NTASKS is set, the default is its value."
        echo "     Otherwise, the default is based on the number of NUMA nodes."
        echo "     The current default is $nranks."
        echo "  -nodes <N>"
        echo "     Set the number of nodes."
        echo "     This is used to compute the default number of OpenMP threads to use per rank."
        echo "     If the env var SLURM_JOB_NUM_NODES is set, the default is its value."
        echo "     Otherwise, the default is one (1)."
        echo "     The current default is $nnodes."
        echo "  -pre_cmd <command(s)>"
        echo "     One or more commands to run before YASK executable."
        echo "  -post_cmd <command(s)>"
        echo "     One or more commands to run after YASK executable."
        echo "  -log <filename>"
        echo "     Write copy of output to <filename>."
        echo "     Default <filename> is based on stencil, arch, hostname, time-stamp, and process ID."
        echo "     Set to empty string ('') to avoid making a log."
        echo "  -log_dir <dir>"
        echo "     Directory name to prepend to log <filename>."
        echo "     Default is '$logdir'."
        echo "  -v"
        echo "     Shortcut for the following options:"
        echo "       $val"
        echo "     If you want to override any of these values, place them after '-v'."
        echo "  -show_arch"
        echo "     Print the default architecture string and exit."
        echo "  <env-var>=<value>"
        echo "     Set environment variable <env-var> to <value>."
        echo "     Repeat as necessary to set multiple vars."
        echo ""
        echo "  All script args not listed above will be passed to the executable."
        echo ""
        echo "  Canonical command issued based on above options:"
        echo "     ssh <-host option> <-sh_prefix option> sh -c -x '<some system-status cmds>; <-pre_cmd option>; env <env vars> <-mpi_cmd option> <-exe_prefix option> <-exe option> <exe args>; <-post_cmd option>'"
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

    elif [[ "$1" == "-nodes" && -n ${2+set} ]]; then
        nnodes=$2
        shift
        shift

    elif [[ "$1" == "-v" ]]; then
        doval=1
        shift

    elif [[ "$1" =~ ^[A-Za-z0-9_]+= ]]; then

        # Something like FOO=bar sets an env var.
        envs+=" $1"
        shift

    elif [[ "$1" == "--" ]]; then
        shift

        # Pass all remaining options to executable and stop parsing.
        opts+=" $@"
        break

    elif [[ "$1" == "-thread_divisor" ]]; then
        echo "Option '$1' is no longer supported."
        echo "Use '-max_threads', '-outer_threads', and/or '-inner_threads'."
        exit 1

    else
        # Check for existance of some binary options, but don't consume them.
        if [[ "$1" == "-outer_threads" ]]; then
            using_opt_outer_threads=1
        elif [[ "$1" == "-trace" ]]; then
            envs+=" KMP_AFFINITY=verbose"
        fi
        
        # Pass this option to executable.
        if [[ $1 =~ [[:space:]] ]]; then
            opts+=" '$1'"
        else
            opts+=" $1"
        fi
        shift
        
    fi

done                            # parsing options.
echo $invo

# Check required opt (yes, it's an oxymoron).
if [[ -z ${stencil:+x} ]]; then
    echo "error: missing required option: -stencil <name>"
    show_stencils
fi

# Set MPI command default.
if [[ $nranks > 1 ]]; then
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
: ${logfile:=yask.$stencil.$arch.$exe_host.`date +%Y-%m-%d_%H-%M-%S`_p$$.log}
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

# Heuristic to determine if this is an offload kernel.
is_offload=0
if [[ $arch =~ "offload" ]]; then
    is_offload=1
fi

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

# Setup paths to run on specified host.
envs2="LD_LIBRARY_PATH=$bindir/../lib:$LD_LIBRARY_PATH$libpath"
if [[ -n "$host" ]]; then
    sh_prefix="ssh $host $sh_prefix"
    envs2+=" PATH=$PATH"

    nm=1
    while true; do
        echo "Verifying access to '$host'..."
        ping -c 1 $host && ssh $host uname -a && break
        echo "Waiting $nm min before trying again..."
        sleep $(( nm++ * 60 ))
    done
fi

# Set OMP threads to number of cores per rank if not already specified and not special.
if [[ ( $using_opt_outer_threads == 0 ) && ( $arch != "knl" ) && ( $is_offload == 0) ]]; then
    if command -v lscpu >/dev/null; then
        nsocks=`lscpu | awk -F: '/Socket.s.:/ { print $2 }'`
        ncores=`lscpu | awk -F: '/Core.s. per socket:/ { print $2 }'`
        if [[ -n "$nsocks" && -n "$ncores" ]]; then
            mthrs=$(( $nsocks * $ncores * $nnodes / $nranks ))
            opts="-outer_threads $mthrs $opts"
        fi
    fi
fi
echo "Num nodes:" $nnodes | tee -a $logfile
echo "Num MPI ranks:" $nranks | tee -a $logfile
echo "Num MPI ranks per node:" $(( $nranks / $nnodes )) | tee -a $logfile

# Add validation opts to beginning.
if [[ $doval == 1 ]]; then
    opts="$val $opts"
fi

# Commands to capture some important system status and config info for benchmark documentation.
config_cmds="sleep 1; uptime; lscpu; cpuinfo -A; sed '/^$/q' /proc/cpuinfo; cpupower frequency-info; uname -a; $dump /etc/system-release; $dump /proc/cmdline; $dump /proc/meminfo; free -gt; numactl -H; ulimit -a; ipcs -l; env | awk '/YASK/ { print \"env:\", \$1 }'"

# Add settings for offload kernel.
if [[ $is_offload == 1 ]]; then
    config_cmds+="; clinfo -l";
    if [[ $nranks > 1 ]]; then
        envs+=" I_MPI_OFFLOAD_TOPOLIB=level_zero I_MPI_OFFLOAD=2"
    else
        envs+=" EnableImplicitScaling=1"
    fi
fi

# Command sequence to be run in a shell.
exe_str="$mpi_cmd $exe_prefix $exe $opts"
cmds="cd $dir; ulimit -s unlimited; $config_cmds; ldd $exe; date; $pre_cmd; env $envs $envs2 $exe_str"
if [[ -n "$post_cmd" ]]; then
    cmds+="; $post_cmd"
fi
cmds+="; date"

echo "===================" | tee -a $logfile

# Finally, invoke the binary in a shell.
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
        echo "Run './utils/bin/yask_log_to_csv.pl $logfile' to output in CSV format."
    fi
    exit $1
}

# Print invocation again.
echo $invo
binvo="Binary invocation: $envs $exe_str"
echo $binvo | tee -a $logfile

# Return a non-zero exit condition if test failed.
if [[ `grep -c 'TEST FAILED' $logfile` > 0 ]]; then
    echo YASK did not pass internal validation test. | tee -a $logfile
    finish 1
fi

# Return a non-zero exit condition if executable didn't exit cleanly.
if [[ `grep -c 'YASK DONE' $logfile` == 0 ]]; then
    echo YASK did not exit cleanly. | tee -a $logfile
    finish 1
fi

# Print a message if test passed on at least one rank.
# (Script would have exited above if any rank failed.)
if [[ `grep -c 'TEST PASSED' $logfile` == $nranks ]]; then
    echo YASK passed internal validation test. | tee -a $logfile
fi

# Print a final message, which will print if not validated or passed validation.
echo YASK ran successfully. | tee -a $logfile
finish 0

