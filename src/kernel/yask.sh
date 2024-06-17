#!/bin/bash

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2024, Intel Corporation
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
envs=""
#envs+=" OMP_DISPLAY_ENV=VERBOSE KMP_VERSION=1"
envs+=" OMP_PLACES=cores KMP_HOT_TEAMS_MODE=1 KMP_HOT_TEAMS_MAX_LEVEL=3"

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
arch_is_def_cpu=1  # arch has been set for CPU default only.

# Default nodes.
nnodes=1
if [[ -n "$SLURM_NNODES" ]]; then
    nnodes=$SLURM_NNODES
fi

# Default type and number of GPUs.
ngpus=0
arch_offload="offload"
if command -v xpu-smi >/dev/null; then
    arch_offload="$def_arch.offload-spir64"
    ngpus=`xpu-smi topology -m | grep -c '^GPU'`
elif command -v nvidia-smi >/dev/null; then
    arch_offload="$def_arch.offload-nv"
    ngpus=`nvidia-smi topo -m | grep -c '^GPU'`
fi
is_offload=0

# Get NUMA nodes.
# The goal is to count only NUMA nodes with CPUs.
# (Systems with HBM may have NUMA nodes without CPUs.)
if command -v numactl >/dev/null; then
    nnumas=`numactl -s | awk '/^cpubind:/ { print NF-1 }'`
elif command -v lscpu >/dev/null; then
    nnumas=`lscpu | grep -c '^NUMA node.*CPU'`
fi

# Default MPI ranks.
# Try Slurm var, then num NUMA nodes.
nranks=$nnodes
nranks_is_def_cpu=1  # nranks has been set for CPU default only.
if [[ -n "$SLURM_NTASKS" && $SLURM_NTASKS > $nnodes ]]; then
    nranks=$SLURM_NTASKS
    nranks_is_def_cpu=0
elif [[ -n "$nnumas" ]]; then
    nranks=$(( $nnumas * $nnodes ))
fi
nranks_offload=$nnodes
if [[ $ngpus > 0 ]]; then
    nranks_offload=$(( $ngpus * $nnodes ))
fi
force_mpi=0

# Other defaults.
pre_cmd=":"  # "colon" command is a no-op.
post_cmd=":"
helping=0
opts=""
bindir=`dirname $0`
logdir="./logs"
tmplog="/tmp/yask-p$$"
dodry=0

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
        echo "     Overrides the default architecture determined from /proc/cpuinfo flags"
        echo "     or the GPU system manangement software for offload kernels."
        echo "     Should correspond to arch=<name> used during compilation"
        echo "     with '.offload-<offload_arch>' appended if built with 'offload=1',"
        echo "     or YK_ARCH=<name> if that was used to override the default."
        echo "     In any case, the '-stencil' and '-arch' args required to launch"
        echo "     any executable are printed at the end of a successful compilation."
        echo "     The default for this host is '$def_arch' for CPU kernels"
        echo "     and '$arch_offload' for offload kernels."
        echo "  -offload"
        echo "     Use an offloaded YASK kernel executable, built with 'offload=1'."
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
        echo "     If -mpi_cmd and -exe_prefix are both specified, this one is used first."
        echo "     The default command is based on the number of nodes and ranks (see below)."
        echo "  -force_mpi"
        echo "     Generate a default 'mpirun' prefix even if there is only 1 rank to run."
        echo "  -nodes <N>"
        echo "     Set the number of nodes."
        echo "     If the env var SLURM_NNODES is set, the default is its value."
        echo "     Otherwise, the default is one (1)."
        echo "     The current default is $nnodes."
        echo "  -ranks <R>"
        echo "     Run the YASK executable on <R> MPI ranks."
        echo "     This value, along with the number of nodes, <N>, is used to set these defaults:"
        echo "      - Number of MPI ranks per node to <R>/<N>."
        echo "      - Number of OpenMP threads per rank based on core count (for CPU kernels only)."
        echo "      - Default MPI command to 'mpirun -np <R> -ppn <R>/<N>'."
        echo "     If a different MPI command is needed, use -mpi_cmd <command> explicitly."
        echo "     If the env var SLURM_NTASKS is set AND if it greater than the number of nodes,"
        echo "        the default is its value."
        echo "     Otherwise, the default is based on the number of NUMA nodes on the current host"
        echo "        for CPU kernels or the number of GPUs on the current host for offload kernels."
        echo "     The current default is $nranks for CPU kernels and $nranks_offload for offload kernels."
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
        echo "     Adds '/tests' to path of log_dir."
        echo "     If you want to override any of these values, place them after '-v'."
        echo "  -show_arch"
        echo "     Print the default architecture string and exit."
        echo "  -dry_run"
        echo "     Do everything except run the YASK executable."
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
        nranks_is_def_cpu=0
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
        arch_is_def_cpu=0
        if [[ $arch =~ "offload" ]]; then
            is_offload=1
        fi
        shift
        shift

    elif [[ "$1" == "-offload" ]]; then
        is_offload=1
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
        nranks_is_def_cpu=0
        shift
        shift

    elif [[ "$1" == "-force_mpi" ]]; then
        force_mpi=1
        shift

    elif [[ "$1" == "-nodes" && -n ${2+set} ]]; then
        nnodes=$2
        shift
        shift

    elif [[ "$1" == "-v" ]]; then
        doval=1
        shift

    elif [[ "$1" == "-dry_run" ]]; then
        dodry=1
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
if [[ -z "$stencil" ]]; then
    echo "error: missing required option: -stencil <name>"
    show_stencils
fi

# Offload settings.
if [[ $is_offload == 1 ]]; then

    # Heuristics for MPI ranks for offload.
    if [[ $nranks_is_def_cpu == 1 ]]; then
        nranks=$nranks_offload
    fi

    # Heuristics for offload arch.
    if [[ $arch_is_def_cpu == 1 ]]; then
        arch=$arch_offload
    fi
fi

# Set MPI command default.
ppn=$(( $nranks / $nnodes ))
if [[ $nranks > 1 || $force_mpi == 1 ]]; then
    : ${mpi_cmd="mpirun -np $nranks -ppn $ppn"}

    # Add default Intel MPI settings.
    envs+=" I_MPI_PRINT_VERSION=1 I_MPI_DEBUG=5"

    # Add NUMA pinning if number of discovered NUMA nodes
    # equals what is being used.
    if [[ -n "$nnumas" && $nnumas == $ppn ]]; then
        envs+=" I_MPI_PIN_DOMAIN=numa"
    fi

    # Check whether HBM policy setting is allowed.
    if [[ `I_MPI_HBW_POLICY=hbw_preferred,hbw_preferred mpirun -np 1 /bin/date |& grep -c 'Unknown memory policy'` == 0 ]]; then
        envs+=" I_MPI_HBW_POLICY=hbw_preferred,hbw_preferred"
    fi
fi

# Bail on errors past this point, but only errors
# in this script, not in the executed commands.
set -e

# Actual host.
exe_host=${host:-`hostname`}

# Command to dump a file to stdout.
dump="head -v -n -0"

# Init log file.
: ${logfile:=yask.$stencil.$arch.$exe_host.n$nnodes.r$nranks.`date +%Y-%m-%d_%H-%M-%S`_p$$.log}
if [[ -n "$logdir" ]]; then
    if [[ $doval == 1 ]]; then
        logdir="$logdir/tests"
    fi
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

# Output some vars.
echo "Num nodes:" $nnodes | tee -a $logfile
echo "Num MPI ranks:" $nranks | tee -a $logfile
echo "Num MPI ranks per node:" $ppn | tee -a $logfile
echo "sh_prefix='$sh_prefix'" | tee -a $logfile
echo "mpi_cmd='$mpi_cmd'" | tee -a $logfile
echo "exe_prefix='$exe_prefix'" | tee -a $logfile
echo "exe='$exe'" | tee -a $logfile
echo "pre_cmd='$pre_cmd'" | tee -a $logfile
echo "post_cmd='$post_cmd'" | tee -a $logfile

# Dump most recent reports.
if [[ -e $make_report ]]; then
    $dump $make_report >> $logfile
    if  [[ -e $yc_report ]]; then
        $dump $yc_report >> $logfile
    fi
fi

# Add validation opts to beginning.
if [[ $doval == 1 ]]; then
    opts="$val $opts"
fi

# Commands to capture some important system status and config info for benchmark documentation.
config_cmds="sleep 1"
if command -v module >/dev/null; then
    config_cmds+="; module list"
fi
config_cmds+="; echo 'Selected env vars:'; env | sort | awk '/YASK|SLURM|SBATCH|MPI|OMP/ { print \"env:\", \$1 }'"

config_cmds+="; set -x" # Start echoing commands before running them.
config_cmds+="; uptime; uname -a; ulimit -a; $dump /etc/system-release; $dump /proc/cmdline; $dump /proc/meminfo; free -gt"
if [[ -r /etc/system-release ]]; then
    config_cmds+="; $dump /etc/system-release"
fi
if command -v lscpu >/dev/null; then
    config_cmds+="; lscpu"
fi
if command -v cpuinfo >/dev/null; then
    config_cmds+="; cpuinfo -A"
fi
if command -v cpupower >/dev/null; then
    config_cmds+="; cpupower frequency-info"
fi
config_cmds+="; sed '/^$/q' /proc/cpuinfo"
if command -v numactl >/dev/null; then
    config_cmds+="; numactl -H"
fi
if command -v ipcx >/dev/null; then
    config_cmds+="; ipcs -l"
fi

# Add settings for offload kernel.
if [[ $is_offload == 1 ]]; then
    if command -v clinfo >/dev/null; then
        config_cmds+="; clinfo -l";
    fi
    if command -v xpu-smi >/dev/null; then
        config_cmds+="; xpu-smi discovery; xpu-smi topology -m";
    fi
    if command -v nvidia-smi >/dev/null; then
        config_cmds+="; nvidia-smi";
    fi
    if [[ -n "$mpi_cmd" ]]; then
        envs+=" I_MPI_OFFLOAD=2"
    fi
fi

# Command sequence to be run in a shell.
exe_str="$mpi_cmd $exe_prefix $exe $opts"
cmds="cd $dir; ulimit -s unlimited; $config_cmds; ldd $exe; date; $pre_cmd; env $envs $envs2 $exe_str; $post_cmd; date"

# Finally, invoke the binary in a shell.
if [[ $dodry == 0 ]]; then
    echo "===================" | tee -a $logfile
    if [[ -z "$sh_prefix" ]]; then
        sh -c "$cmds" 2>&1 | tee -a $logfile
    else
        echo "Running shell under '$sh_prefix'..."
        $sh_prefix "sh -c '$cmds'" 2>&1 | tee -a $logfile
    fi
    echo "===================" | tee -a $logfile
fi

# Exit if just getting help.
if [[ $helping == 1 ]]; then
    exit 0
fi

function finish {
    if [[ "$logfile" == $tmplog ]]; then
        rm $tmplog
    else
        echo "Log saved in '$logfile'."
        if [[ $dodry == 0 ]]; then
            echo "Run './utils/bin/yask_log_to_csv.pl $logfile' to output in CSV format."
        fi
    fi
    exit $1
}

# Print invocation again.
echo $invo
binvo="Binary invocation: $envs $exe_str"
echo $binvo | tee -a $logfile

# Return if dry-run.
if [[ $dodry == 1 ]]; then
    echo YASK not started due to -dry_run option. | tee -a $logfile
    finish 0
fi

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

