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

# Purpose: provide common utility functions for YASK.

package YaskUtils;

use strict;
use FileHandle;
use Carp;
use Scalar::Util qw(looks_like_number);

# Values to get from log file.
# First one should be overall "fitness".
# Key must start with the following string.
# Case ignored and spaces and hyphens interchangeable.
our @log_keys =
  (
   # values from binary.
   'mid throughput (num-points/sec)',
   'mid throughput (num-reads/sec)',
   'mid throughput (num-writes/sec)',
   'mid throughput (est-FLOPS)',
   'mid elapsed time (sec)',
   'mid num-steps-done',

   'best throughput (num-points/sec)',
   'best throughput (num-reads/sec)',
   'best throughput (num-writes/sec)',
   'best throughput (est-FLOPS)',
   'best elapsed time (sec)',
   'best num-steps-done',

   'num-trials',
   'min-throughput (num-points/sec)',
   'max-throughput (num-points/sec)',
   'ave-throughput (num-points/sec)',
   'std-dev-throughput (num-points/sec)',

   'stencil name',
   'stencil description',
   'element size',
   'script invocation',
   'binary invocation',
   'yask version',
   'target',

   'num nodes',
   'num MPI ranks',
   'num MPI ranks per node',
   'num OpenMP threads', # also matches 'Num OpenMP threads used'.
   'num outer threads',
   'num inner threads',
   'device thread limit',

   'domain size in this rank',
   'total allocation in this rank',
   'overall problem size',
   'total overall allocation',
   'inner-layout dim',
   'inner-loop dim',

   'num mega-blocks per local-domain per step',
   'num blocks per mega-block per step',
   'num micro-blocks per block per step',
   'num nano-blocks per micro-block per step',
   'num pico-blocks per nano-block per step',

   'L1 prefetch distance',
   'L2 prefetch distance',
   'num temporal block steps',
   'num wave front steps',
   'extra padding',
   'min padding',

   # values from compiler report
   'YASK compiler invocation',
   'YC_STENCIL',
   'YC_TARGET',
   'YK_CXXVER',
   'YK_CXXCMD',
   'YK_CXXOPT',
   'YK_CXXFLAGS',
   'YK_STENCIL',
   'YK_ARCH',
   'YK_TAG',
   'YK_EXEC',
 );

# Keys set with custom code.
my $linux_key = "Linux kernel";
my $hostname_key = "hostname";
my $nodes_key = "MPI node(s)";
my $auto_tuner_key = "Auto-tuner used";
my $val_key = "validation results";
my $yask_key = "YASK env vars";
our @special_log_keys =
  (
   $hostname_key,
   $linux_key,
   $nodes_key,
   $auto_tuner_key,
   $val_key,
   $yask_key,
  );

#  Sizes.
our @size_log_keys =
  (
   'global-domain size',
   'local-domain size',
   'mega-block size',
   'block size',
   'micro-block size',
   'nano-block size',
   'pico-block size',
   'vector size',
  );
if (0) {
  push @size_log_keys,
    (
     'local-domain tile size',
     'mega-block tile size',
     'block tile size',
     'micro-block tile size',
     'nano-block tile size',
     );
}

# System settings.
our @sys_log_keys =
  (
   'model name',
   'CPU(s)',
   'core(s) per socket',
   'socket(s)',
   'NUMA node(s)',
   'MemTotal',
   'MemFree',
   'ShMem',
  );

our @all_log_keys = ( @log_keys, @size_log_keys, @sys_log_keys, @special_log_keys );

our $oneKi = 1024;
our $oneMi = $oneKi * $oneKi;
our $oneGi = $oneKi * $oneMi;
our $oneTi = $oneKi * $oneGi;
our $onePi = $oneKi * $oneTi;
our $oneEi = $oneKi * $onePi;
our $oneK = 1e3;
our $oneM = 1e6;
our $oneG = 1e9;
our $oneT = 1e12;
our $oneP = 1e15;
our $oneE = 1e18;
our $onem = 1e-3;
our $oneu = 1e-6;
our $onen = 1e-9;
our $onep = 1e-12;
our $onef = 1e-15;

# Return a number from a number w/suffix.
# Examples:
# - removeSuf("2.34K") => 2340.
# - removeSuf("2KiB") => 2048.
# - removeSuf("34 M") => 34000000.
# - removeSuf("foo") => "foo".
sub removeSuf($) {
  my $val = shift;

  # Not starting w/a number?
  return $val if $val !~ /^[0-9]/;

  # Not containing a number followed by a letter?
  return $val if $val !~ /[0-9]\s*[A-Za-z]/;

  # Look for suffix.
  if ($val =~ /^([0-9.e+-]+)B$/i) {
    $val = $1;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Ki?B$/i) {
    $val = $1 * $oneKi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Mi?B$/i) {
    $val = $1 * $oneMi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Gi?B$/i) {
    $val = $1 * $oneGi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Ti?B$/i) {
    $val = $1 * $oneTi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Pi?B$/i) {
    $val = $1 * $onePi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*Ei?B$/i) {
    $val = $1 * $oneEi;
  } elsif ($val =~ /^([0-9.e+-]+)K$/) {
    $val = $1 * $oneK;
  } elsif ($val =~ /^([0-9.e+-]+)M$/) {
    $val = $1 * $oneM;
  } elsif ($val =~ /^([0-9.e+-]+)G$/) {
    $val = $1 * $oneG;
  } elsif ($val =~ /^([0-9.e+-]+)T$/) {
    $val = $1 * $oneT;
  } elsif ($val =~ /^([0-9.e+-]+)P$/) {
    $val = $1 * $oneP;
  } elsif ($val =~ /^([0-9.e+-]+)E$/) {
    $val = $1 * $oneE;
  } elsif ($val =~ /^([0-9.e+-]+)f$/) {
    $val = $1 * $onef;
  } elsif ($val =~ /^([0-9.e+-]+)p$/) {
    $val = $1 * $onep;
  } elsif ($val =~ /^([0-9.e+-]+)n$/) {
    $val = $1 * $onen;
  } elsif ($val =~ /^([0-9.e+-]+)u$/) {
    $val = $1 * $oneu;
  } elsif ($val =~ /^([0-9.e+-]+)m$/) {
    $val = $1 * $onem;
  }
  return $val;
}

# set one or more results from one line of output.
my %proc_keys;
my $klen = 6;
sub getResultsFromLine($$) {
  my $results = shift;          # ref to hash.
  my $line = shift;             # 1 line of output.

  chomp($line);

  # pre-process keys one time.
  if (scalar keys %proc_keys == 0) {
    undef %proc_keys;
    for my $k (@log_keys, @size_log_keys, @sys_log_keys) {

      my $pm = lc $k;
      $pm =~ s/^\s+//;
      $pm =~ s/\s+$//;

      # relax hyphen and space match by converting all hyphen-space
      # sequences to single hyphens.
      $pm =~ s/[- ]+/-/g;

      # short key.
      my $sk = substr $pm,0,$klen;

      # escape regex chars.
      $pm =~ s/\(/\\(/g;
      $pm =~ s/\)/\\)/g;

      $proc_keys{$sk}{$pm} = $k;
    }
  }

  # Substitutions to handle old formats.
  $line =~ s/^Invocation/Script invocation/g if $line !~ /yask_compiler/;
  $line =~ s/overall.problem/global-domain/g; # Doesn't change "Overall problem size in N rank(s)"
  $line =~ s/rank.domain/local-domain/g;
  $line =~ s/grid/var/g;
  $line =~ s/Grid/Var/g;
  $line =~ s/target.ISA/target/g;
  $line =~ s/mini([_-])bl/micro${1}bl/g;
  $line =~ s/sub([_-])bl/nano${1}bl/g;
  $line =~ s/region/mega-block/g;
  $line =~ s/minimum-padding/min-padding/g;
  $line =~ s/Num threads per region/num outer threads/g;
  $line =~ s/Num threads per block/num inner threads/g;
  
  # special cases for manual parsing...

  # Validation.
  if ($line =~ /did not pass internal validation test/i) {
    $results->{$val_key} = 'failed';
  }
  elsif ($line =~ /passed internal validation test/i) {
    $results->{$val_key} = 'passed';
  }
  elsif ($line =~ /Results NOT VERIFIED/i) {
    $results->{$val_key} = 'not verified';
  }
  
  # Output of 'uname -a'
  elsif ($line =~ /^\s*Linux\s/) {
    my @w = split ' ', $line;

    # 'Linux' hostname kernel ...
    $results->{$hostname_key} = $w[1];
    $results->{$linux_key} = $w[2];
  }

  # MPI node names.
  # [0] MPI startup(): 0       97842    epb333     {0,1,2,3,4,...
  elsif ($line =~ /MPI startup\(\):\s*\d+\s+\d+\s+(\S+)/) {
    my $nname = $1;
    $results->{$nodes_key} .= ' ' if defined $results->{$nodes_key};
    $results->{$nodes_key} .= $nname;
  }

  # Vars containing "YASK".
  elsif ($line =~ /^env:\s+(\w*YASK\w*=.*)/) {
    $results->{$yask_key} .= '; ' if
      exists $results->{$yask_key};
    $results->{$yask_key} .= $1;
  }

  # If auto-tuner is run globally, capture updated values.
  # Invalidate settings overridden by auto-tuner on multiple stages.
  elsif ($line =~ /^\s*auto-tuner(.).*size:/) {
    my $c = $1;
    $results->{$auto_tuner_key} = 'TRUE';

    # If colon found immediately after "auto-tuner", tuner is global.
    my $onep = ($c eq ':');
    
    for my $k (@size_log_keys) {
      $line =~ s/-size/ size/;
      if ($line =~ / (best-)?$k:\s*(.*)/i) {
        my $val = $onep ? $2 : 'auto-tuned';
        $results->{$k} = $val;
      }
    }
  }

  # look for matches to all other keys.
  else {
    my ($key, $val) = split /[=:]/,$line,2;
    if (defined $val) {

      # make canonical version of key.
      $key = lc $key;
      $key =~ s/^\s+//;
      $key =~ s/\s+$//;
      $key =~ s/[- ]+/-/g;      # relax hyphen and space match.

      # trim value.
      $val =~ s/^\s+//;
      $val =~ s/\s+$//;

      # short key for quick match.
      my $sk = substr $key,0,$klen;

      # return if no match to short key.
      return if !exists $proc_keys{$sk};

      # Look for matches to each full key for the given short key.
      # Only compares key to beginning of target,
      # so compare longer keys first.
      # Example: must compare "Num MPI ranks per node" before "Num MPI ranks".
      for my $m (sort { length($b) <=> length($a) } keys %{$proc_keys{$sk}}) {

        # match?
        if ($key =~ /^$m/) {
          $val =~ s/^\s+//;
          $val =~ s/\s+$//;
          $val = removeSuf($val);

          # Save value w/converted suffix.
          my $k = $proc_keys{$sk}{$m};
          $results->{$k} = $val;

          # More special processing to get env-vars set via script.
          # TODO: remove overridden vars.
          if ($k eq 'script invocation') {
            for my $w (split /\s+/, $val) {
              if ($w =~ /(\w*YASK\w*=.*)/) {
                $results->{$yask_key} .= '; ' if
                  exists $results->{$yask_key};
                $results->{$yask_key} .= $1;
              }
            }
          }
          last;                 # stop after first match.
        }
      }
    }
  }
}

# Parse given log file.
# Fill in hash of results.
sub getResultsFromFile($$) {
  my $results = shift;          # ref to hash.
  my $fname = shift;            # filename.

  my $fh = new FileHandle;
  if (!$fh->open("<$fname")) {
    carp "error: cannot open '$fname'";
  } else {
    while (<$fh>) {
      getResultsFromLine($results, $_);
    }
    $fh->close();
  }
}

# Print standard CSV header to given file.
# Does NOT print newline.
sub printCsvHeader($) {
  my $fh = shift;               # file handle.

  print $fh join(',', @all_log_keys);
}

# Print hash values to given file.
# Does NOT print newline.
sub printCsvValues($$) {
  my $results = shift;          # ref to hash.
  my $fh = shift;               # file handle.

  my @cols;
  for my $m (@all_log_keys) {
    my $r = $results->{$m};

    $r = '' if
      !defined $r;

    # special-case fix for bogus Excel cell reference.
    $r =~ s/-/ -/ if
      $r =~ /^"?-[a-zA-Z]/;

    # add quotes if not a number, etc.
    $r = '"'.$r.'"' if
      !looks_like_number($r) &&
      $r !~ /^"/ &&
      $r ne 'TRUE' && $r ne 'FALSE';
    push @cols, $r;
  }
  print $fh join(',', @cols);
}

# return with a 1 so require() will not fail.
1;
