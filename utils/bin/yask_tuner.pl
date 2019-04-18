#!/usr/bin/env perl

##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2019, Intel Corporation
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

# Purpose: Use a Genetic Algorithm to explore workload compile and/or run options.
# Or, sweep through all possible settings if -sweep option is given.

use strict;
use File::Basename;
use File::Path;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";
use AI::Genetic;
use English;
use FileHandle;
use Text::ParseWords;
use Sys::Hostname;
use List::Util qw(first max maxstr min minstr reduce shuffle sum);
use YaskUtils;
use POSIX;
use Digest::MD5 qw(md5 md5_hex md5_base64);

# constants.
my @dirs = qw(x y z);           # not including t.

# command-line options.
my $outDir = 'logs';           # dir for output.
my $stencil;                   # type of stencil.
my $arch;                      # target architecture.
my $sweep = 0;                 # if true, sweep instead of search.
my $testing = 0;               # if true, don't run real trials.
my $checking = 0;              # if true, don't run at all.
my $mic;                       # set to 0, 1, etc. for KNC mic.
my $host;                      # set to run on a different host.
my $sde = 0;                   # run under sde (for testing).
my $sim = 0;                   # run under any simulator/emulator.
my $radius;                    # stencil radius.
my $zVec = 0;                  # Force 1D vectorization in 'z' direction.
my $folding = 1;               # 2D & 3D folding allowed.
my $dp;                        # double precision.
my $makeArgs = '';             # extra make arguments.
my $makePrefix = '';           # prefix for make.
my $makeTimeout = 60 * 10;     # max secs for make to run.
my $runArgs = '';              # extra run arguments.
my $maxGB = 32;                # max mem usage.
my $minGB = 0;                 # min mem usage.
my $nranks;                    # num ranks.
my $debugCheck = 0;            # print each initial check result.
my $doBuild = 1;               # do compiles.
my $doVal = 0;                 # do validation runs.
my $maxVecsInCluster = 4;      # max vectors in a cluster.
my @folds = ();                # folding variations to explore.
my $killCmd = '/usr/bin/timeout'; # command to kill runaway jobs.

# Set default arch.
$arch = `bash src/kernel/yask.sh -show_arch`;
chomp $arch;

sub usage {
  my $msg = shift;              # error message or undef.

  warn "error: $msg\n" if defined $msg;
  warn
      "usage: $0 [options]\n".
      "\nhigh-level control options:\n".
      " -val               Run validation before each performance test.\n".
      " -outDir            Directory to write output into (default is '$outDir').\n".
      " -check             Print the settings, sanity-check the initial population, and exit.\n".
      " -test              Run search with dummy fitness values.\n".
      " -debugCheck        Print detailed results of sanity-checks.\n".
      " -killCmd=<PATH>    Command to kill runaway jobs (default is '$killCmd').\n".
      "\ntarget options:\n".
      " -arch=<ARCH>       Specify target architecture: knc, knl, hsw, ... (default is '$arch').\n".
      " -host=<NAME>       Run binary on host <NAME> using ssh.\n".
      " -mic=<N>           Set hostname to current hostname appended with -mic<N>; sets arch to 'knc'.\n".
      " -sde               Run binary on SDE (for testing only).\n".
      " -makePrefix=<CMD>  Prefix make command with <CMD>.*\n".
      " -makeArgs=<ARGS>   Pass additional <ARGS> to make command.*\n".
      " -makeTimeout=<N>   Max secs to allow make to run (default is $makeTimeout).\n".
      " -runArgs=<ARGS>    Pass additional <ARGS> to bin/yask.sh command.\n".
      " -ranks=<N>         Number of ranks to use on host (default uses setting in 'yask.sh').\n".
      "\nstencil options:\n".
      " -stencil=<NAME>    Specify stencil: iso3dfd, awp, etc. (required).\n".
      " -dp|-sp            Specify FP precision (default is SP).*\n".
      " -radius=<N>        Specify stencil radius for stencils that use this option (default is 8).*\n".
      "\nsearch-space options:\n".
      " -noBuild           Do not compile kernel; kernel binary must already exist.\n".
      "                    Disables all build-related genes.\n".
      " -sweep             Use exhausitive search instead of GA.\n".
      " -<gene_name>=<N>   Force <gene_name> to fixed value <N>.\n".
      "                    Run with -check for list of genes and default ranges.\n".
      "                    Setting local-domain size (l) also sets upper block and region sizes.\n".
      "                    Leave off 'x', 'y', 'z' suffix to set these 3 vars to same val.\n".
      "                    Examples: '-l=512'      Set local-domain size to 512^3.\n".
      "                              '-bx=64'      Set block size to 64 in 'x' dim.\n".
      "                              '-ep=0'       Disable extra padding.\n".
      "                              '-c=1'        Allow only one vector in a cluster.\n".
      "                              '-r=0'        Allow only one OpenMP region (region size=0 => local-domain size).\n".
      " -<gene_name>=<N>-<M> Restrict <gene_name> between <N> and <M>, inclusive.\n".
      "                    Example:  '-bx=8-128'.\n".
      "                    See the notes above on <gene_name> specification.\n".
      " -<gene_name>=<N>-<M>:<S>  Restrict <gene_name> between <N> and <M> with stride <S>.\n".
      "                    Example:  '-bx=8-128:4'.\n".
      "                    See the notes above on <gene_name> specification.\n".
      " -folds=<list>      Comma separated list of folds to use.*\n".
      "                    Examples: '-folds=4 4 1', '-folds=1 1 16, 4 4 1, 1 4 4'.\n".
      "                    Can only specify 3D folds.\n".
      " -mem=<N>-<M>       Set allowable est. memory usage between <N> and <M> GiB (default is $minGB-$maxGB).\n".
      "                    Ignored if all problem-size vars are set to a fixed value.\n".
      " -maxVecsInCluster=<N>  Maximum vectors allowed in cluster (default is $maxVecsInCluster).*\n".
      " -noPrefetch        Disable any prefetching (shortcut for '-pfd_l1=0 -pfd_l2=0').*\n".
      " -noFolding         Allow only 1D vectorization (in any direction).*\n".
      " -zVec              Force 1D vectorization in 'z' direction (traditional 'inline' style).*\n".
      "\n".
      "* indicates options that are invalid if -noBuild is used.\n".
      "\n".
      "examples:\n".
      " $0 -stencil=iso3dfd -l=768 -r=0 -noPrefetch\n".
      " $0 -stencil=awp -lx=512 -ly=512 -lz=256 -b=4-512:4\n".
      " $0 -stencil=3axis -mem=8-10 -noBuild\n";

  exit(defined $msg ? 1 : 0);
}

# Restrict gene $geneRanges{key}
# to $geneRanges{key}[0]
# or $geneRanges{key}[0]-$geneRanges{key}[1]
# or $geneRanges{key}[0]-$geneRanges{key}[1] by $geneRanges{key}[2].
my %geneRanges;
my $autoKey = 'def_';          # prefix for default setting.

# control groups.
# TODO: make an option.
my $showGroups = 0;

# autoflush.
$| = 1;

# process args.
my $invo = join(' ', map { s/\"/\\\"/g; '"'.$_.'"' } $0, @ARGV);
print "Invocation: $invo\n";
for my $origOpt (@ARGV) {
  my $opt = lc $origOpt;
  my ($lhs, $rhs) = split /=/, $origOpt, 2;

  if ($opt eq '-h' || $opt eq '-help') {
    usage();
  }
  elsif ($opt eq '-test') {
    $testing = 1;
  }
  elsif ($opt eq '-check') {
    $checking = 1;
  }
  elsif ($opt eq '-debugcheck') {
    $debugCheck = 1;
  }
  elsif ($opt =~ /^-?killcmd=(.*)$/i) {
    $killCmd = $rhs;
  }
  elsif ($opt =~ /^-?outdir=(.*)$/i) {
    $outDir = $rhs;
  }
  elsif ($opt =~ /^-?makeargs=(.*)$/i) {
    $makeArgs = $rhs;
  }
  elsif ($opt =~ /^-?makeprefix=(.*)$/i) {
    $makePrefix = $rhs;
  }
  elsif ($opt =~ '^-?maketimeout=(\d+)$') {
    $makeTimeout = $rhs;
  }
  elsif ($opt =~ /^-?runargs=(.*)$/i) {
    $runArgs = $rhs;
  }
  elsif ($opt eq '-sde') {
    $sde = 1;
    $sim = 1;
  }
  elsif ($opt =~ '^-?ranks=(\d+)$') {
    $nranks = $rhs;
  }
  elsif ($opt =~ '^-?mic=(\d+)$') {
    $mic = $rhs;
    $arch = 'knc';
    $host = hostname()."-mic$mic";
  }
  elsif ($opt =~ '^-?arch=(\S+)$') {
    $arch = $rhs;
  }
  elsif ($opt =~ /^-?host=(\S+)$/) {
    $host = $rhs;
  }
  elsif ($opt eq '-dp') {
    $dp = 1;
  }
  elsif ($opt eq '-sp') {
    $dp = 0;
  }
  elsif ($opt =~ '^-?mem=([.\d]+)-([.\d]+)$') {
    $minGB = $1;
    $maxGB = $2;
  }
  elsif ($opt =~ '^-?radius=(\d+)$') {
    $radius = $rhs;
  }
  elsif ($opt =~ '^-?stencil=(\w+)$') {
    $stencil = $rhs;
  }
  elsif ($opt eq '-noprefetch') {
    $geneRanges{$autoKey.'pfd_l1'} = [ 0 ];
    $geneRanges{$autoKey.'pfd_l2'} = [ 0 ];
  }
  elsif ($opt =~ '^-?maxvecsincluster=(\d+)$') {
    $maxVecsInCluster = $rhs;
  }
  elsif ($opt eq '-zvec') {
    $zVec = 1;
    $folding = 0;
  }
  elsif ($opt eq '-nofolding') {
    $folding = 0;
  }
  elsif ($opt eq '-nobuild') {
    $doBuild = 0;
  }
  elsif ($opt eq '-val') {
    $doVal = 1;
  }
  elsif ($opt eq '-sweep') {
    $sweep = 1;
    print "Sweeping all values instead of searching with GA.\n";
  }
  elsif ($opt =~ '^-?folds=(\s*\d+\s+\d+\s+\d+\s*(,\s*\d+\s+\d+\s+\d+\s*)*)$') {
    my $val = $rhs;
    $val =~ tr/ //s;
    $val =~ s/^\s+|\s+$//g;
    $val =~ s/,\s+/,/g;
    $val =~ s/\s+,/,/g;
    @folds = split(',',$val);
  }

  # Assume a gene name if nothing else matches.
  elsif ($opt =~ /^-?(\S+)=(\d+)(-(\d+))?(:(\d+))?$/) {
    my ($key, $min, $max, $stride) = ($1, $2, $4, $6);
    $max = $min if !defined $max;
    $stride = 1 if !defined $stride;
    $geneRanges{$key} = [ $min, $max, $stride ];
    usage("min value $min for '$key' > max value $max.")
      if ($min > $max);

    # special case for local-domain size: also set default for other max sizes.
    if ($key =~ /^l[xyz]?$/ && $max > 0) {
      my @szs = qw(r b mb sb);
      push @szs, qw(bg mbg sbg) if $showGroups;
      for my $i (@szs) {
        my $key2 = $key;
        $key2 =~ s/^l/$i/;
        $geneRanges{$autoKey.$key2} = [ 1, $max ];
      }
    }
  }
  else {
    usage("unrecognized option '$opt' (be sure to use '=' for options that need a value).");
  }
}

usage("must specify stencil.") if !defined $stencil;
usage("must specify arch.") if !defined $arch;

# precision.
my $realBytes = $dp ? 8 : 4;

# vector.
my $vbits = ($arch =~ /^(k|sk)/) ? 512 : 256;
my $velems = $vbits / 8 / $realBytes;

# radius.
$radius = 8 if !defined $radius;

# disable folding for DP MIC (no valignq).
$folding = 0 if (defined $mic && $dp);

# dir name.
my $searchTypeStr = $sweep ? 'sweep' : 'tuner';
my $hostStr = defined $host ? $host : hostname();
my $timeStamp=`date +%Y-%m-%d_%H-%M`;
chomp $timeStamp;
my $baseName = "yask_$searchTypeStr.$stencil.$arch.$hostStr.${timeStamp}_p$$";
$outDir = '.' if !$outDir;
$outDir .= "/$baseName";
print "Output will be saved in '$outDir'.\n";

# open output.
mkpath($outDir,1);
if (!$checking) {
  my $outTxtFile = "$outDir/$baseName.txt";
  my $outTFH = new FileHandle;
  $outTFH->open(">$outTxtFile") or die "error: cannot write to '$outTxtFile'\n";
  print $outTFH "$invo\n";
  $outTFH->close();
}

my $outFile = "$outDir/$baseName.csv";
my $outFH = new FileHandle;
$outFH->open(">$outFile") or die "error: cannot write to '$outFile'\n"
  unless $checking;

# things to get from the run.
if ($showGroups) {
  push @YaskUtils::log_keys,
    'block-group-size',
    'mini-block-group-size';
    'sub-block-group-size';
}

# how many individuals to create randomly and then keep at any given time.
my $popSize = 200;

# experiment ends when any one of the following is reached.
my $numGens = 1000;           # stop when this many generations are run. Set to 1 for purely random search.
my $maxOkRuns = 10000;        # stop when this many individuals are successfully run.
my $maxFlatGens = 4;          # stop when this many generations do not improve the fitness.

my $secPerTrial = 60;           # used to estimate time required.

# prefetching.
my $maxPfd_l1 = 4;
my $maxPfd_l2 = 8;

# dimension-related vars.
my $minDim = 128;        # min dimension on any axis.
my $maxDim = 2 * $YaskUtils::oneKi;  # max dimension on any axis.
my $maxPad = 3;
my $maxTimeBlock = 10;          # max temporal blocking.
my $maxCluster = 4;
my $minPoints;
my $maxPoints;
my $minClustersInBlock = 10;
my $minBlocksInRegion = 10;

# Threads.
my $minThreadDivisor = 1;
my $maxThreadDivisor = 4;
my $minBlockThreads = 1;
my $maxBlockThreads = 32;       # TODO: set to number of CPUs.

# List of possible loop orders.
my @loopOrders =
  ('123', '132', '213', '231', '312', '321');

# Possible space-filling curve modifiers.
my @pathNames =
  ('', 'serpentine', 'square_wave serpentine', 'grouped');

# List of folds.
if ( !@folds ) {

  # start with inline in z only.
  @folds = "1 1 $velems";

  # add more 1D options if not z-vec.
  push @folds, ("1 $velems 1", "$velems 1 1") if !$zVec;

  # add remaining options if folding.
  push @folds, ($velems == 8) ?
    ("4 2 1", "4 1 2",
     "2 4 1", "2 1 4",
     "1 4 2", "1 2 4") :
    ($velems == 16) ?
    ("8 2 1", "8 1 2",
     "4 4 1", "4 2 2", "4 1 4",
     "2 8 1", "2 4 2", "2 2 4", "2 1 8",
     "1 8 2", "1 4 4", "1 2 8") :
    # velems == 4
    ("2 2 1", "2 1 2", "1 2 2")
    if $folding;
}

# OMP.
my @schedules =
  ( 'static,1', 'dynamic,1', 'guided,1' );

# Data structure to describe each gene in the genome.
# 2-D array. Each outer array element contains the following elements:
# 0. min allowed value; '0' is a special case handled by YASK.
# 1. max allowed value.
# 2. step size between values (usually 1).
# 3. name.
my @rangesAll = 
  (
   # rank size.
   [ $minDim, $maxDim, 16, 'lx' ],
   [ $minDim, $maxDim, 16, 'ly' ],
   [ $minDim, $maxDim, 16, 'lz' ],

   # region size.
   [ 1, $maxTimeBlock, 1, 'rt' ],
   [ 0, $maxDim, 1, 'rx' ],
   [ 0, $maxDim, 1, 'ry' ],
   [ 0, $maxDim, 1, 'rz' ],

   # block size.
   [ 1, $maxTimeBlock, 1, 'bt' ],
   [ 0, $maxDim, 1, 'bx' ],
   [ 0, $maxDim, 1, 'by' ],
   [ 0, $maxDim, 1, 'bz' ],

   # mini-block size.
   [ 0, $maxDim, 1, 'mbx' ],
   [ 0, $maxDim, 1, 'mby' ],
   [ 0, $maxDim, 1, 'mbz' ],

   # sub-block size.
   [ 0, $maxDim, 1, 'sbx' ],
   [ 0, $maxDim, 1, 'sby' ],
   [ 0, $maxDim, 1, 'sbz' ],

   # extra padding.
   [ 0, $maxPad, 1, 'epx' ],
   [ 0, $maxPad, 1, 'epy' ],
   [ 0, $maxPad, 1, 'epz' ],

   # threads.
   [ $minThreadDivisor, $maxThreadDivisor, 1, 'thread_divisor' ],
   [ $minBlockThreads, $maxBlockThreads, 1, 'block_threads' ],
   [ 0, 1, 1, 'bind_block_threads' ],
  );

if ($showGroups) {
  push @rangesAll,
    (
     # block-group size.
     [ 0, $maxDim, 1, 'bgx' ],
     [ 0, $maxDim, 1, 'bgy' ],
     [ 0, $maxDim, 1, 'bgz' ],
     
     # sub-block-group size.
     [ 0, $maxDim, 1, 'sbgx' ],
     [ 0, $maxDim, 1, 'sbgy' ],
     [ 0, $maxDim, 1, 'sbgz' ],
    );
}

# Add compiler genes.
if ($doBuild) {
  push @rangesAll,
    (

     # Loops, from the list above.
     # Each loop consists of index order and path mods.
     # There is no var for rank or block path because they
     # must strictly increase.
     [ 0, $#loopOrders, 1, 'subBlockOrder' ],
     [ 0, $#pathNames, 1, 'subBlockPath' ],
     [ 0, $#loopOrders, 1, 'miniBlockOrder' ],
     [ 0, $#pathNames, 1, 'miniBlockPath' ],
     [ 0, $#loopOrders, 1, 'blockOrder' ],
     [ 0, $#loopOrders, 1, 'regionOrder' ],
     [ 0, $#pathNames, 1, 'regionPath' ],
     [ 0, $#loopOrders, 1, 'rankOrder' ],

     # how to shape vectors, from the list above.
     [ 0, $#folds, 1, 'fold' ],

     # vector-cluster sizes.
     [ 1, $maxCluster, 1, 'cx' ],
     [ 1, $maxCluster, 1, 'cy' ],
     [ 1, $maxCluster, 1, 'cz' ],

     # prefetch distances for l1 and l2.
     # all non-pos numbers => no prefetching, so ~50% chance of being enabled.
     [ -$maxPfd_l1, $maxPfd_l1, 1, 'pfd_l1' ],
     [ -$maxPfd_l2, $maxPfd_l2, 1, 'pfd_l2' ],

     # other build options.
     [ 0, $#schedules, 1, 'ompRegionSchedule' ], # OMP schedule for region loop.
     [ 0, $#schedules, 1, 'ompBlockSchedule' ], # OMP schedule for block loop.

    );
}

# indices for each element of @rangesAll.
my $minI = 0;
my $maxI = 1;
my $stepI = 2;
my $nameI = 3;
my $numIs = 4;

# Genes w/only one value.
# These will be removed from the GA search.
my %fixedVals;

# Fill in default values in dynamic gene ranges.
for my $key (keys %geneRanges) {

  # cleanup.
  my ($min, $max, $step) = @{$geneRanges{$key}};
  $max = $min if !defined $max;
  $step = 1 if !defined $step;
  $geneRanges{$key} = [ $min, $max, $step ];
}

# Apply dynamic gene ranges and make some checks on the final ranges.
my %usedGeneRanges;
for my $i (0..$#rangesAll) {
  my $r = $rangesAll[$i];        #  ref to array.
  my $key = $r->[$nameI];

  # Look for match in dynamic limits.
  # Try full-match of key, then w/o x,y,z suffix.
  # Try each w/cmd-line specified key, then w/auto-gen key.
  # E.g., for 'lx', try the following: 'lx', 'l', 'auto_lx', 'auto_l'.
  my $rkey = lc $key;
  my $rkey2 = $rkey;
  $rkey2 =~ s/[xyz]$//;          # e.g., 'lx' -> 'l'.
  for my $rk ($rkey, $rkey2, $autoKey.$rkey, $autoKey.$rkey2) {
    if (exists $geneRanges{$rk}) {
      my ($min, $max, $step) = @{$geneRanges{$rk}};
      $r->[$minI] = $min;
      $r->[$maxI] = $max;
      $r->[$stepI] = $step;
      last;
    }
  }

  # Mark all forms as used.
  $usedGeneRanges{$rkey} = 1;
  $usedGeneRanges{$rkey2} = 1;
  $usedGeneRanges{$autoKey.$rkey} = 1;
  $usedGeneRanges{$autoKey.$rkey2} = 1;

  # Check range.
  die "internal error: not $numIs elements in ranges[$key]\n"
    unless scalar @{$r} == $numIs;
  die "error: key: max value $r->[$maxI] of '$key' < min value $r->[$minI]\n"
    if $r->[$minI] > $r->[$maxI];

  # Add to fixed-vals if only one choice.
  if ($r->[$minI] == $r->[$maxI]) {
    $fixedVals{$key} = $r->[$minI];
    print "Gene '$key' set to $fixedVals{$key}.\n";
  }
}

# disable memory range if all domain sizes fixed.
if ((scalar grep { exists $fixedVals{"l$_"} } @dirs) == scalar @dirs) {
  undef $minGB;
  undef $maxGB;
}

# check that all ranges were used.
for my $key (keys %geneRanges) {
  die "error: gene '$key' not recognized.\n"
    unless exists $usedGeneRanges{$key};
}

# remove fixed values from search.
my @ranges = grep { my $key = $_->[$nameI];
                    !exists $fixedVals{$key} } @rangesAll;

# keep track of GA progress.
my $run = 0;
my $okRuns = 0;
my $bestFit = 0;
my $bestGen = 0;

sub init() {
  $run = 0;
  $okRuns = 0;
  $bestFit = 0;
  $bestGen = 0;
}

# convert a number n so that values in [a0..a1] are mapped to [b0..b1],
# values < a0 => b0.
# values > a1 => b1.
sub adjRange($$$$$) {
  my ($n, $a0, $a1, $b0, $b1) = @_;

  # outside input range?
  if ($n <= $a0) {
    return $b0;
  } elsif ($n >= $a1) {
    return $b1;
  }

  # value ranges.
  my $ar = $a1 - $a0;
  my $br = $b1 - $b0;

  $n -= $a0;                    # 0..ar.
  $n /= $ar;                    # 0..1
  $n *= $br;                    # 0..br.
  $n += $b0;                    # b0..b1

  # convert to int and double-check range.
  $n = int($n);
  $n = $b0 if $n < $b0;
  $n = $b1 if $n > $b1;

  return $n;
}

# print the vars.
sub printValues($) {
  my $values = shift;
  for my $i (0..$#$values) {
    print "  $ranges[$i][$nameI] = $values->[$i]\n";
  }
}

# make a hash to access genes by name.
# input: ref to list of genes.
# output: ref to hash of genes by name.
sub makeHash($) {
  my $values = shift;
  
  my %h;
  for my $i (0..$#ranges) {
    $h{$ranges[$i][$nameI]} = $values->[$i];
  }

  return \%h;
}

# Get value in this preference order:
# 1. hash parameter.
# 2. fixed values from cmd line.
sub readHash($$$) {
  my $hash = shift;
  my $key = shift;
  my $isBuildVar = shift;

  # try hash.
  my $val = $hash->{$key};
  return $val if defined $val;

  # try fixed cmd-line vals.
  $val = $fixedVals{$key};
  return $val if defined $val;

  # return dummy value for build var if disabled.
  return 1 if (!$doBuild && $isBuildVar);

  die "internal error: value for gene '$key' not provided.\n";
}

# Call readHash across 3 directions and return array.
sub readHashes($$$) {
  my $hash = shift;
  my $key = shift;
  my $isBuildVar = shift;

  my @vals;
  for my $d (@dirs) {
    if ($key =~ /bg$/ && !$showGroups) {
      push @vals, 1;
    } else {
      push @vals, readHash($hash, "$key$d", $isBuildVar);
    }
  }
  return @vals;
}

# multiply args.
sub mult {
  
  my $n = 1;
  map { $n *= $_ } @_;
  return $n;
}

sub roundUp($$) {
  my $n = shift;
  my $mult = shift;
  return int(ceil($n / $mult) * $mult);
}

# round the first arg to a multiple of the second arg.
# make sure result is between third and forth arg.
sub roundToMult($$$$) {
  my $v = shift;
  my $m = shift;
  my $min = shift;
  my $max = shift;

  my $val = int(($v + $m/2) / $m) * $m;
  $val = $min if $val < $min;
  $val = $max if $val > $max;

  return $val;
}

# round all values to those given in ranges.
sub roundValuesToMult($) {
  my $values = shift;           # ref to array.

  for my $i (0..$#ranges) {
    $values->[$i] = roundToMult($values->[$i], $ranges[$i][$stepI],
                                $ranges[$i][$minI], $ranges[$i][$maxI]);
  }
}

# create the basic make command and stencil tag.
sub getMakeCmd($$) {
  my $macros = shift;
  my $margs = shift;

  my $tag = $stencil;
  my $makeCmd = "echo 'build disabled'";

  if ($doBuild) {
    my $tagPrefix = $tag."_p".$$."_";
    $tag = $tagPrefix.md5_hex($macros, $margs, $makeArgs, $realBytes, $radius);

    # Remove binaries that haven't been used in a while.
    system "find bin lib -name '*$tagPrefix*' -amin +15 | xargs --no-run-if-empty rm";

    # Already exists?
    if (-x "bin/yask_kernel.$tag.$arch.exe" &&
        -x "lib/libyask_kernel.$tag.$arch.so") {
      $makeCmd = "echo 'binary & library exist'";
    }
    else {
      $makeCmd =
        "$makePrefix make -j EXTRA_MACROS='$macros' YK_STENCIL=$tag ".
        "stencil=$stencil arch=$arch real_bytes=$realBytes radius=$radius $margs $makeArgs";
      $makeCmd = "$makeCmd default; $makeCmd clean";
    }
  }

  return ( $makeCmd, $tag );
}

# create the basic run command.
my $timeCmd = 'time';
sub getRunCmd($) {
  my $tag = shift;
  
  my $exePrefix = $timeCmd;
  $exePrefix .= " sde -$arch --" if $sde;

  my $runCmd = "bin/yask.sh -log $outDir/yask.$stencil.$arch.run".sprintf("%06d",$run).".log";
  if (defined $mic) {
    $runCmd .= " -mic $mic";
  } else {
    $runCmd .= " -host $host" if defined $host;
  }
  $runCmd .= " -exe_prefix '$exePrefix' -stencil $tag -arch $arch -no-pre_auto_tune -no-print_suffixes";
  $runCmd .= " -ranks $nranks" if defined $nranks;
  return $runCmd;
}

# return estimate of mem footprint in bytes.
my $numSpatialGrids = 0;
sub calcSize($$$) {
  my $sizes = shift;            # ref to size array.
  my $pads = shift;             # ref to pad array.
  my $mults = shift;            # ref to array of multiples.

  # need to determine how many XYZ grids will be allocated for this stencil.
  # TODO: get info from compiler report.
  if (!$numSpatialGrids) {

    my ( $makeCmd, $tag ) = getMakeCmd('', 'EXTRA_CXXFLAGS=-O1');
    my $runCmd = getRunCmd($tag)." -t 0 -l 32 $runArgs";
    my $cmd = "$makeCmd 2>&1 && $runCmd";

    my $timeDim = 0;
    my $numGrids = 0;
    my $numUpdatedGrids = 0;
    my @cmdOut;
    if ($testing) {
      $numSpatialGrids = 1;
    } else {
      print "Running '$cmd' to determine number of grids...\n";
      open CMD, "$cmd 2>&1 |" or die "error: cannot run '$cmd'\n";
      while (<CMD>) {
        chomp;
        my $line = $_;
        push @cmdOut, $line;

        # E.g.,
        # 'A' 4-D var (t=2 * x=8 * y=48 * z=49) with storage at 0x2aba63016000 ...
        my $ngrids = 1;
        if (/\d-?D .*x=.*y=.*z=/) {
          for my $w (split ' ',$line) {
            if ($w =~ /(\w+)=(\d+)/) {
              my ($dim, $sz) = ($1, $2);
              if ($dim eq 't') {
                $ngrids *= $sz;
              }
            }
          }
          $numSpatialGrids += $ngrids;
          print "$line => $ngrids XYZ grids\n";
        }
      }
      close CMD;
    }
    if (!$numSpatialGrids) {
      map { print ">> $_\n"; } @cmdOut;
      die "error: no relevant grid allocations found in '$cmd'; $0 only works with 'x, y, z' 3-D stencils.\n";
    }
    print "Determined that $numSpatialGrids XYZ grids are allocated.\n";
  }

  # estimate each dim of allocated memory as size + 2 * (halo + pad).
  my @sizes = map { roundUp($sizes->[$_], $mults->[$_]) +
                      2 * roundUp($radius + $pads->[$_], $mults->[$_]) } 0..$#dirs;

  my $n = mult(@sizes);      # mult sizes plus padding & halos.
  my $nb = $n * $realBytes;

  # mult by number of grids.
  $nb *= $numSpatialGrids;

  return $nb;
}

# calculate fitness from results hash.
sub results2fitness($) {
  my $results = shift;
  my $fitness = 0;

  my $fitnessMetric = $YaskUtils::log_keys[0]; # first one is fitness.

  # simple lookup.
  if (defined $results->{$fitnessMetric}) {
    $fitness = $results->{$fitnessMetric};
  } else {
    print "WARNING: not fitness result found.\n";
  }

  # any adjustments can go here.

  print "fitness = $fitness\n";
  return $fitness;
}

# check success from one line of output.
sub setPassed($$) {
  my $passed = shift;          # ref to var.
  my $line = shift;             # 1 line of output.

  $$passed = 1 if (/TEST PASSED/);
}


# hash of previous results, keyed by command to run.
my %resultsCache;
my %fitnessCache;
my %testCache;

# run the command and return fitness and various associated data in a hash.
sub evalIndiv($$$$$$) {
  my $makeCmd = shift;
  my $testCmd = shift;
  my $simCmd = shift;
  my $shortRunCmd = shift;
  my $longRunCmd = shift;
  my $pts = shift;

  # check result cache.
  my $rkey = "$makeCmd; $longRunCmd";
  my $fitness = 0;
  my $results = $resultsCache{$rkey};
  if (defined $results) {
    print "using cached results for '$rkey'.\n";
    $fitness = $fitnessCache{$rkey};
    return $fitness, $results;
  }
  $results = {};

  # check test cache.
  my $tkey = "$makeCmd; $testCmd";
  my $passed = 0;
  if ($doVal && $testCmd) {
    my $testRes = $testCache{$tkey};
    if (defined $testRes) {
      print "using cached test for '$tkey'.\n";
      if (!$testRes) {
        print "aborting because previous test failed.\n";
        return $fitness, $results;
      }
      $passed = 1;                # indicate already passed.
    }
  } else {
    $passed = 1;                # no test to run.
  }

  # just make a nonsense fitness value if testing script.
  if ($testing) {
    $fitness = md5($simCmd);
    $passed = 1;
  } else {
    my $made = 0;

    # do several runs:
    # 0=test
    # if not sim,
    #  1=short
    #  2=long
    # else
    #  1=sim
    for my $N (0..2) {

      # already tested?
      next if $N == 0 && $passed;

      # which run?
      my $cmd = ($N == 0) ? $testCmd :
        $sim ? $simCmd :
          ($N == 1) ? $shortRunCmd :
            $longRunCmd;

      # need to make?
      if (!$made) {
        $cmd = "$makeCmd 2>&1 && $cmd";
        $made = 1;
      }

      my $secs;
      print "running '$cmd' ...\n";
      open CMD, "$cmd 2>&1 |" or die "error: cannot run '$cmd'\n";
      while (<CMD>) {
        print ">> $_";

        # test run.
        if ($N == 0) {
          setPassed(\$passed, $_);
        }

        # perf run.
        else {

          # collect run-time from short run only.
          # try several different formats from 'time' prefix.
          if ($N == 1) {

            # 0.01user 0.02system 1:19.93elapsed 0%CPU (0avgtext+0avgdata 14784maxresident)k
            if (/(\d+):([.\d]+)elapsed/) {
              $secs = $1 * 60 + $2;
              print "elapsed time is $secs secs.\n";
            }

            # 0.000u 0.024s 0:00.03 66.6%     0+0k 0+0io 0pf+0w
            elsif (/\ds\s+(\d+):([.\d]+)/) {
              $secs = $1 * 60 + $2;
              print "elapsed time is $secs secs.\n";
            }

            # real    0m9.905s
            elsif (/^real\s+(\d+)m([.\d]+)s/) {
              $secs = $1 * 60 + $2;
              print "elapsed time is $secs secs.\n";
            }
          }

          # look for expected metrics in output.
          YaskUtils::getResultsFromLine($results, $_);
        }
      }
      close CMD;

      # calc fitness from results.
      $fitness = results2fitness($results);

      # bail if not passed.
      if (!$passed) {
        print "stopping because test did not pass\n";
        last;
      }

      # checks for short run.
      if ($N == 1) {

        # bail for simulation.
        last if $sim;

        # bail if fitness not close to best.
        if ($fitness < $bestFit * 0.9) {
          print "stopping after short run due to non-promising fitness\n";
          last;
        } else {
          print "short run looks promising; continuing with long run...\n";
        }
      }
    } # N
  }

  $testCache{$tkey} = $passed;
  $resultsCache{$rkey} = $results;
  $fitnessCache{$rkey} = $fitness;
  return $fitness, $results;
}

# return loop-ctrl vars.
sub makeLoopVars($$$$$) {
  my $h = shift;
  my $makePrefix = shift;       # e.g., 'BLOCK'.
  my $tunerPrefix = shift;      # e.g., 'block'.
  my $reqdMods = shift;         # e.g., ''.
  my $lastDim = shift;          # e.g., 2 or 3.

  my $order = readHash($h, $tunerPrefix."Order", 1);
  my $orderStr = $loopOrders[$order];           # e.g., '231'.

  # dimension vars.
  my @dims = split '',$orderStr;      # e.g., ('2', '3', '1).
  @dims = grep { $_ <= $lastDim } @dims; # e.g., ('2', '1).

  # vars to create.
  my $order = join(',', @dims);  # e.g., '2, 1'.
  my $outerMods = $reqdMods;
  my $innerMods = '';

  # path gene?
  my $pathKey = $tunerPrefix."Path";
  if (exists $h->{$pathKey}) {
    my $path = readHash($h, $pathKey, 1);
    my $pathStr = @pathNames[$path];                # e.g., 'grouped'.
    $outerMods = "$pathStr $outerMods";
  }
  
  my $loopVars = " ".$makePrefix."_LOOP_ORDER='$order'";
  $loopVars .= " ".$makePrefix."_LOOP_OUTER_MODS='$outerMods'";
  $loopVars .= " ".$makePrefix."_LOOP_INNER_MODS='$innerMods'";
  return $loopVars;
}

# sanity-check vars.
my $numChecks = 0;
my %checkStats;
my %checkNums;
my %checkSums;
my %checkMins;
my %checkMaxs;
my $justChecking = 1;           # set to 0 to do actual run.
sub addStat($$$) {
  my ($ok, $name, $val) = @_;
  $name = $ok ? "check-passed $name" : "check-failed $name";
  $checkNums{$name}++;
  $checkSums{$name} += $val;
  $checkMins{$name} = $val if !defined $checkMins{$name} || $val < $checkMins{$name};
  $checkMaxs{$name} = $val if !defined $checkMaxs{$name} || $val > $checkMaxs{$name};
}

sub printCheckStats {
  print "Sanity-check stats:\n";
  for my $k (sort keys %checkStats) {
    print "  $k: $checkStats{$k}\n";
  }
  print "  total checked: $numChecks\n";
  if (exists $checkStats{'ok'}) {
    my $n = $checkStats{'ok'};
    print "  fraction ok: ".($n / $numChecks)."\n";
    for my $k (sort keys %checkSums) {
      my $ave = $checkNums{$k} ? $checkSums{$k} / $checkNums{$k} : 0;
      print "  $k: num=$checkNums{$k}, min=$checkMins{$k}, ave=$ave, max=$checkMaxs{$k}\n";
    }
  }
}

# adjust inner sizes to fit into outer sizes.
sub adjSizes($$) {
  my $is = shift;               # ref to inner sizes.
  my $os = shift;               # ref to outer sizes.
  
  # adjust each dim.
  map {

    # If outer size is zero, do nothing.
    if ($os->[$_] == 0) {
    }
    
    # If size is zero, set to max of outer.
    elsif ($is->[$_] == 0) {
      $is->[$_] = $os->[$_];
    }

    # Wrap around.
    # TODO: change from abrupt wrap-around function to
    # one w/o discontinuities, e.g., /\/\/\/\ instead of /|/|/|/|/|.
    else {
      $is->[$_] = (($is->[$_] - 1) % $os->[$_]) + 1;

      # Bump up to outer size if close.  This is a heuristic to avoid tiny
      # remainders.  It also gives a higher probability of exactly matching
      # the outer size.
      if ($is->[$_] > $os->[$_] * 0.9) {
        $is->[$_] = $os->[$_];
      }
    }

  } 0..$#dirs;
}

# if just checking, return 0 or 1.
# else, return the goodness of an individual; higher is better.
# fitness values should be > 0.
sub fitness {
  my $values = shift;           # ref to array.
  roundValuesToMult($values);

  my $ok = 1; # set to 0 if found to be bad.

  my $gen = int($run / $popSize) + 1;
  my $indiv = ($run % $popSize) + 1;
  if (!$justChecking) {
    $run++;
    print "======= run $run: individual $indiv in generation $gen =======\n";
  }
  if ($debugCheck || !$justChecking) {
    print "Settings to be evaluated:\n";
    printValues($values);
  }

  # get individual vars from hash or fixed values.
  my $h = makeHash($values);
  my @ds = readHashes($h, 'l', 0);
  my $rt = readHash($h, 'rt', 1);
  my @rs = readHashes($h, 'r', 0);
  my $bt = readHash($h, 'bt', 1);
  my @bs = readHashes($h, 'b', 0);
  my @mbs = readHashes($h, 'mb', 0);
  my @sbs = readHashes($h, 'sb', 0);
  my @bgs = readHashes($h, 'bg', 0);
  my @sbgs = readHashes($h, 'sbg', 0);
  my @cvs = readHashes($h, 'c', 1); # in vectors, not in points!
  my @ps = readHashes($h, 'ep', 0);
  my $fold = readHash($h, 'fold', 1);
  my $thread_divisor = readHash($h, 'thread_divisor', 0);
  my $block_threads = readHash($h, 'block_threads', 0);
  my $bind_block_threads = readHash($h, 'bind_block_threads', 0);
  my $pfd_l1 = readHash($h, 'pfd_l1', 1);
  my $pfd_l2 = readHash($h, 'pfd_l2', 1);
  my $ompRegionSchedule = readHash($h, 'ompRegionSchedule', 1);
  my $ompBlockSchedule = readHash($h, 'ompBlockSchedule', 1);

  # fold numbers.
  my $foldNums = $folds[$fold];
  my @fs = split ' ', $foldNums;

  # vectors in cluster.
  my $cvs = mult(@cvs);
  if ($cvs > $maxVecsInCluster) {
    print "  overall cluster size of $cvs vectors > $maxVecsInCluster\n" if $debugCheck;
    $checkStats{'cluster too large'}++;
    $ok = 0;
  }

  # cluster sizes in points.
  my @cs = map { $fs[$_] * $cvs[$_] } 0..$#dirs;

  # adjust inner sizes to fit in their enclosing sizes.
  adjSizes(\@rs, \@ds);         # region <= domain.
  adjSizes(\@bs, \@rs);         # block <= region.
  adjSizes(\@mbs, \@bs);        # mini-block <= block.
  adjSizes(\@sbs, \@mbs);       # sub-block <= mini-block.
  adjSizes(\@bgs, \@rs);        # block-group <= region.
  adjSizes(\@sbgs, \@mbs);      # sub-block-group <= mini-block.

  # 3d sizes in points.
  my $dPts = mult(@ds);
  my $rPts = mult(@rs);
  my $bPts = mult(@bs);
  my $mbPts = mult(@mbs);
  my $sbPts = mult(@sbs);
  my $cPts = mult(@cs);
  my $fPts = mult(@fs);
  my $bgPts = mult(@bgs);
  my $sbgPts = mult(@sbgs);

  # Clusters per block.
  my @bcs = map { ceil($bs[$_] / $cs[$_]) } 0..$#dirs;
  my $bCls = mult(@bcs);

  # Mini-blocks per block.
  my @bmbs = map { ceil($bs[$_] / $mbs[$_]) } 0..$#dirs;
  my $bMbs = mult(@bmbs);

  # Blocks per region.
  my @rbs = map { ceil($rs[$_] / $bs[$_]) } 0..$#dirs;
  my $rBlks = mult(@rbs);

  # Regions per rank.
  my @drs = map { ceil($ds[$_] / $rs[$_]) } 0..$#dirs;
  my $dRegs = mult(@drs);

  # mem usage estimate.
  my $overallSize = calcSize(\@ds, \@ps, \@cs);

  if ($debugCheck) {
    print "Sizes:\n";
    print "  local-domain size = $dPts\n";
    print "  region size = $rPts\n";
    print "  block size = $bPts\n";
    print "  sub-block size = $sbPts\n";
    print "  cluster size = $cPts\n";
    print "  fold size = $fPts\n";
    print "  regions per local-domain = $dRegs\n";
    print "  blocks per region = $rBlks\n";
    print "  clusters per block = $bCls\n";
    print "  mini-blocks per block = $bMbs\n";
    print "  mem estimate = ".($overallSize/$YaskUtils::oneGi)." GB\n";
    if ($showGroups) {
      print "  block-group size = $bgPts\n";
      print "  sub-block-group size = $sbgPts\n";
    }
  }

  # check overall size.
  if (defined $minGB && $overallSize / $YaskUtils::oneGi < $minGB) {
    print "  overall size of $overallSize bytes < $minGB GiB\n" if $debugCheck;
    $checkStats{'mem too low'}++;
    $ok = 0;
  } elsif (defined $maxGB && $overallSize / $YaskUtils::oneGi > $maxGB) {
    print "  overall size of $overallSize bytes > $maxGB GiB\n" if $debugCheck;
    $checkStats{'mem too high'}++;
    $ok = 0;
  }

  # check points.
  if (defined $minPoints && $dPts < $minPoints) {
    print "  $dPts points is too low\n" if $debugCheck;
    $checkStats{'num points too low'}++;
    $ok = 0;
  } elsif (defined $maxPoints && $dPts > $maxPoints) {
    print "  $dPts points is too high\n" if $debugCheck;
    $checkStats{'num points too high'}++;
    $ok = 0;
  }

  # Each block should do minimal work.
  if ($bCls < $minClustersInBlock) {
    print "  $bCls clusters per block < $minClustersInBlock\n" if $debugCheck;
    $checkStats{'block size too small'}++;
    $ok = 0;
  }

  # Should be min number of blocks.
  if ($rBlks < $minBlocksInRegion) {
    print "  $rBlks blocks per region < $minBlocksInRegion\n" if $debugCheck;
    $checkStats{'too few blocks per region'}++;
    $ok = 0;
  }

  # all sanity checks done.
  print "Sanity check passed\n" if $ok && $debugCheck;
  $numChecks++;
  $checkStats{'ok'} += $ok;
  addStat($ok, 'mem estimate', $overallSize);
  addStat($ok, 'local-domain size', $dPts);
  addStat($ok, 'region size', $rPts);
  addStat($ok, 'block size', $bPts);
  addStat($ok, 'mini-block size', $mbPts);
  addStat($ok, 'sub-block size', $sbPts);
  addStat($ok, 'cluster size', $cPts);
  addStat($ok, 'regions per local-domain', $dRegs);
  addStat($ok, 'blocks per region', $rBlks);
  addStat($ok, 'clusters per block', $bCls);
  addStat($ok, 'mini-blocks per block', $bMbs);
  addStat($ok, 'vectors per cluster', $cvs);
  if ($showGroups) {
    addStat($ok, 'block-group size', $bgPts);
    addStat($ok, 'sub-block-group size', $sbgPts);
  }

  # exit here if just checking.
  return $ok if $justChecking;

  # OMP settings.
  my $regionScheduleStr = $schedules[$ompRegionSchedule];
  my $blockScheduleStr = $schedules[$ompBlockSchedule];

  # compile-time settings.
  my $macros = '';              # string of macros.
  my $mvars = '';               # other make vars.

  # prefetch distances.
  $pfd_l1 = 0 if $pfd_l1 < 0;
  $pfd_l2 = 0 if $pfd_l2 < 0;
  if ($pfd_l1 > 0 && $pfd_l2 > 0) {

    # make sure pfld2 > pfld1.
    $pfd_l2 = $pfd_l1 + 1 if $pfd_l1 >= $pfd_l2;
  }
  $mvars .= " pfd_l1=$pfd_l1 pfd_l2=$pfd_l2";

  # cluster & fold.
  $mvars .= " cluster=x=$cvs[0],y=$cvs[1],z=$cvs[2]";
  $mvars .= " fold=x=$fs[0],y=$fs[1],z=$fs[2]";

  # gen-loops vars.
  $mvars .= makeLoopVars($h, 'RANK', 'rank', '', 3);
  $mvars .= makeLoopVars($h, 'REGION', 'region', '', 3);
  $mvars .= makeLoopVars($h, 'BLOCK', 'block', '', 3);
  $mvars .= makeLoopVars($h, 'MINI_BLOCK', 'miniBlock', '', 3);
  $mvars .= makeLoopVars($h, 'SUB_BLOCK', 'subBlock', '', 2);

  # other vars.
  $mvars .= " omp_region_schedule=$regionScheduleStr omp_block_schedule=$blockScheduleStr";

  # how to make.
  my ( $makeCmd, $tag ) = getMakeCmd($macros, $mvars);

  # how to run.
  my $runCmd = getRunCmd($tag);     # shell command plus any initial args.
  my $args = "";             # exe args.
  $args .= " -thread_divisor $thread_divisor";
  $args .= " -block_threads $block_threads";
  $args .= ($bind_block_threads ? " " : " -no"). "-bind_block_threads";

  # sizes.
  $args .= " -lx $ds[0] -ly $ds[1] -lz $ds[2]";
  $args .= " -rt $rt -rx $rs[0] -ry $rs[1] -rz $rs[2]";
  $args .= " -bt $bt -bx $bs[0] -by $bs[1] -bz $bs[2]";
  $args .= " -mbx $mbs[0] -mby $mbs[1] -mbz $mbs[2]";
  $args .= " -sbx $sbs[0] -sby $sbs[1] -sbz $sbs[2]";
  $args .= " -epx $ps[0] -epy $ps[1] -epz $ps[2]";
  if ($showGroups) {
    $args .= " -bgx $bgs[0] -bgy $bgs[1] -bgz $bgs[2]";
    $args .= " -sbgx $sbgs[0] -sbgy $sbgs[1] -sbgz $sbgs[2]";
  }

  # num of secs and trials.
  my $shortTime = 1;
  my $shortTrials = 1;
  my $longTime = 5;
  my $longTrials = min($gen, 2);

  # various commands.
  my $testCmd = "$runCmd -v $runArgs"; # validation on a small problem size.
  my $simCmd = "$runCmd $args -t 1 -dt 1 $runArgs";  # simulation w/1 trial & 1 step.
  my $shortRunCmd = "$runCmd $args -t $shortTrials -trial_time $shortTime $runArgs"; # fast run for 'upper-bound' time.
  my $longRunCmd = "$runCmd $args -t $longTrials -trial_time $longTime $runArgs";  # normal run w/more trials.

  # add kill command to prevent runaway code.
  if (-x $killCmd) {

    # for the make command.
    $makeCmd = "$killCmd $makeTimeout $makeCmd";
    
    # for the run commands.
    if (!$sde) {

      # Kill time is some overhead plus allocated time multipled by temporal blocking and num runs.
      my $maxOHead = 60 * 10;
      my $killTime = $maxOHead + ($shortTime * max($rt, $bt) * $shortTrials);

      # Inject kill command into '-exe_prefix' part of run commands.
      $shortRunCmd =~ s/$timeCmd/$timeCmd $killCmd $killTime/;

      # Repeat for long run.
      $killTime = $maxOHead + ($longTime * max($rt, $bt) * $longTrials);
      $longRunCmd =~ s/$timeCmd/$timeCmd $killCmd $killTime/;
    }
  }

  # do actual fitness eval if sanity check passed.
  my $fitness;
  my $results = {};
  if ($ok) {
    ($fitness, $results) = evalIndiv($makeCmd, $testCmd, $simCmd, $shortRunCmd, $longRunCmd, $dPts);
    print "results:\n";
    for my $k (sort keys %$results) {
      print "  $k: $results->{$k}\n";
    }
  }

  # count good runs.
  if (defined $fitness && $fitness > 0) {
    $okRuns++;
  }

  # on failure, set fitness to small number.
  else {
    $fitness = 1e-6;
  }

  # track best result.
  my $isBest = 0;
  if ($fitness > $bestFit) {
    $bestFit = $fitness;
    $bestGen = $gen;
    $isBest = 1;
  }

  # print results to CSV file.
  my @cols = ( $run, $gen, $indiv );
  for my $fk (sort keys %fixedVals) {
    push @cols, $fixedVals{$fk};
  }
  push @cols, @$values, '"'.$makeCmd.'"', '"'.$longRunCmd.'"';
  print $outFH join(',', @cols,'');
  YaskUtils::printCsvValues($results, $outFH);
  @cols = ( $fitness, $bestGen, $bestFit, $isBest ? 'TRUE':'FALSE' );
  print $outFH join(',', '',@cols), "\n";

  print "final fitness = $fitness\n".
    "=====================================\n";
  return $fitness;
}

# called at end of every generation.
# terminate if non-zero returned.
sub terminateFunc {
  my $ga = shift;
  my $best = $ga->getFittest->score();
  my $gen = $ga->generation()+1;

  print "========== end of generation $gen ===============\n";
  printCheckStats();
  print "Best fitness so far (from gen $bestGen) = $best\n";

  if ($okRuns >= $maxOkRuns) {
    print "max runs of $maxOkRuns reached; exiting.\n";
    return 1;
  }

  if ($bestGen + $maxFlatGens < $gen) {
    print "exiting because the best generation was ".($gen - $bestGen)." generations ago.\n";
    return 1;
  }

  return 0;
}

# sweep through var #i; recurse if not last var.
# returns number of runs made.
sub sweep {
  my $doRun = shift;            # whether to actually run.
  my $i = shift;                # index into array.
  my @a = @_;                   # array of current values.

  my $n = 0;
  for (my $v = $ranges[$i][$minI]; $v <= $ranges[$i][$maxI]; $v += $ranges[$i][$stepI]) {
    $a[$i] = $v;

    # evaluate if at end of list.
    if ($i == $#ranges) {
      if (check(\@a)) {
        $n++;
        fitness(\@a) if $doRun;
      }
    }

    # otherwise, recurse.
    else {
      $n += sweep($doRun, $i + 1, @a);
    }
  }
  return $n;
}

sub printNumCombos($) {
  my $nt = shift;
  printf "Total: %.3g trials\n".
    "Time est.: %.2g hrs (%.2g days) assuming $secPerTrial secs per trial\n",
      $nt, $nt * $secPerTrial / 60/60, $nt * $secPerTrial / 60/60/24;
}

# header
my @names = map { $_->[$nameI] } @ranges;
print $outFH join(',', "run", "generation", "individual",
                   sort(keys %fixedVals), @names,
                  "make command", "run command", '');
YaskUtils::printCsvHeader($outFH);
print $outFH join(',', '', "fitness", "best generation so far",
                  "best fitness so far", "is this best so far"), "\n"
 unless $checking;
print "\nSize of search space:\n";
my $nt = 1;
for my $i (0..$#ranges) {
  my $n = int(($ranges[$i][$maxI] - $ranges[$i][$minI]) / $ranges[$i][$stepI]) + 1;
  my $by = ($ranges[$i][$stepI] == 1) ? '' : " by $ranges[$i][$stepI]";
  print "  '$ranges[$i][$nameI]': \t$ranges[$i][$minI] - $ranges[$i][$maxI]$by: \t$n values\n";
  $nt *= $n;
}
printNumCombos($nt);
print "Memory footprint restriction: >= $minGB GiB.\n"
  if defined $minGB;
print "Memory footprint restriction: <= $maxGB GiB.\n"
  if defined $maxGB;

if ($sweep) {
  $popSize = $nt;  # just so all output shows as 1st gen.

  die "search space is too large for an exhaustive sweep; reduce the ranges and retry"
    if $nt > 1e8;

  print "\nNumber of trials passing initial check:\n";
  my @values;
  $nt = sweep(0, 0, @values);
  printNumCombos($nt);

  die "search space is too large for an exhaustive sweep; reduce the ranges and retry"
    if $nt > 1e6;

  die "Exiting due to -check option.\n" if $checking;
  print "\nStarting in 5 sec...\n";
  sleep 5;
  init();
  sweep(1, 0, @values);

} else {

  my $nt = $popSize * $numGens;
  $nt = $maxOkRuns if $maxOkRuns < $nt;
  print "Max number of trials for GA if it does not converge:\n";
  print "  population size = $popSize\n";
  print "  num generations = $numGens\n";
  print "  max evals = $maxOkRuns\n";
  printNumCombos($nt);

  # see http://search.cpan.org/~aqumsieh/AI-Genetic-0.05/Genetic.pm
  # -check option was added, and some bugs were fixed.
  my $ga = new AI::Genetic( -fitness    => \&fitness,
                            -check      => \&fitness,
                            -type       => 'rangevector',
                            -population => $popSize,
                            -crossover  => 0.90,
                            -mutation   => 0.10,
                            -terminate  => \&terminateFunc,
                            -maxCheckFails => 100000,
                          );
  
  print "Creating initial population of $popSize...\n";
  init();
  my $ok = $ga->init(\@ranges);
  printCheckStats();
  die "Exiting due to -check option.\n" if $checking;

  if ($ok) {
    print "Starting evaluations in 5 sec...\n";
    sleep 5;
    $justChecking = 0;
    $debugCheck = 1;
    $ga->evolve('rouletteTwoPoint', $numGens);
  
    print "Size = ", $ga->size(), "\n";
    print "Gen = ", $ga->generation()+1, "\n";
    print "Best score = ", $ga->getFittest->score(), "\n";
  }
}
close $outFH;
print "Done; output in '$outDir'.\n";
