##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2018, Intel Corporation
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

# Values to get from log file.
# Key must start with the following string.
# Case ignored.
# Spaces and hyphens interchangeable.
our @log_keys =
  (
   # values from binary.
   'best throughput (num-points/sec)',
   'best throughput (num-reads/sec)',
   'best throughput (num-writes/sec)',
   'best throughput (est-FLOPS)',
   'best elapsed time (sec)',
   'best num-steps-done',
   'version',
   'stencil name',
   'invocation',
   'binary invocation',
   'num ranks',
   'num OpenMP threads',
   'num threads per region',
   'num threads per block',
   'total overall allocation',
   'overall problem size',
   'rank-domain size',
   'region size',
   'block size',
   'mini-block size',
   'sub-block size',
   'cluster size',
   'vector size',
   'num regions per rank-domain per step',
   'num blocks per region per step',
   'num mini-blocks per block per step',
   'num sub-blocks per mini-block per step',
   'extra padding',
   'minimum padding',
   'L1 prefetch distance',
   'L2 prefetch distance',
   'num temporal block steps',
   'num wave front steps',

   # other values from log file.
   'model name',
   'CPU(s)',
   'core(s) per socket',
   'socket(s)',
   'NUMA node(s)',
   'mem total',
   'mem free',
  );


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
# - removeSuf("2.34K") => 2300.
# - removeSuf("2KiB") => 2048.
# - removeSuf("foo") => "foo".
sub removeSuf($) {
  my $val = shift;

  # Not a number?
  return $val if $val !~ /^[0-9]/;

  # Look for suffix.
  if ($val =~ /^([0-9.e+-]+)\s*(KiB|kB)?$/) {
    $val = $1 * $oneKi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*(MiB|MB)?$/) {
    $val = $1 * $oneMi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*(GiB|GB)?$/) {
    $val = $1 * $oneGi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*(TiB|TB)?$/) {
    $val = $1 * $oneTi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*(PiB|PB)?$/) {
    $val = $1 * $onePi;
  } elsif ($val =~ /^([0-9.e+-]+)\s*(EiB|EB)?$/) {
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
sub getResultsFromLine($$) {
  my $results = shift;          # ref to hash.
  my $line = shift;             # 1 line of output.

  return unless $line =~ /[:=]/;

  # look for expected metrics.
  for my $m (@log_keys) {

    # escape regex chars.
    my $mre = $m;
    $mre =~ s/\(/\\(/g;
    $mre =~ s/\)/\\)/g;

    # hyphen or space can match either or none.
    $mre =~ s/[- ]/\[- \]?/g;

    # look for metric at beginning of line followed by ':' or '='.
    if ($line =~ /^\s*$mre[^:=]*[:=]\s*(.+)/i) {
      my $val = $1;

      # Save value w/converted suffix.
      $results->{$m} = removeSuf($val);
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

  print $fh join(',', @log_keys);
}

# Print hash values to given file.
# Does NOT print newline.
sub printCsvValues($$) {
  my $results = shift;          # ref to hash.
  my $fh = shift;

  my @cols;
  for my $m (@log_keys) {
    my $r = $results->{$m};
    $r = '' if !defined $r;
    $r = '"'.$r.'"' if $r !~ /^[0-9.e+-]+$/;  # add quotes if not a number.
    push @cols, $r;
  }
  print $fh join(',', @cols);
}

# return with a 1 so require() will not fail...
#
1;

