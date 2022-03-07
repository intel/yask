#!/usr/bin/env perl

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2022, Intel Corporation
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

# Purpose: Process the output of a log from a binary and compare every write.
# Build with the following options: real_bytes=8 trace_mem=1
# Run kernel with '-v -force_scalar -trace' and pipe output to this script.

use strict;
use File::Basename;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

my $in_perf = 0;
my $in_val = 0;
my $key = undef;
my @key_stack;
my %vals;
my %pts;
my %writes;

while (<>) {

  if (/Trial number:\s*(\d+)/) {
    $key = ($1 == 1) ? 'perf' : undef;
  }

  elsif (/Running.*for validation/) {
    $key = 'val';
  }

  elsif (/Checking results/) {
    undef $key;
  }

  # suspend collection during halo exchange.
  elsif (/exchange_halos\s*$/) {
    push @key_stack, $key;
    undef $key;
  }  
  elsif (/exchange_halos: secs spent in this call/) {
    $key = pop @key_stack;
  }

  # writeElem: pressure[t=0, x=0, y=0, z=0] = 5.7 at line 287
  elsif (/^(YASK: )?writeElem:\s*(\w+)\[(.*)\]\s*=\s*(\S+)/) {
    my ($var, $indices, $val) = ($2, $3, $4);
    if (defined $key) {
      $indices =~ s/\b\d\b/0$&/g; # make indices 2 digits.

      # track last value.
      $vals{$key}{$var}{$indices} = $val;

      # track all pts written.
      $pts{$var}{$indices} = 1;

      # track writes in order.
      push @{$writes{$key}}, [ $var, $indices, $val ];
    }
  }

  elsif (/PASS|FAIL|DONE/) {
    print $_;
  }
}
my $nissues = 0;

sub comp($$$) {
  my $var = shift;
  my $indices = shift;
  my $pval = shift;
  
  print "$var\[$indices\] =";
  if (defined $pval) {
    print "\t $pval";
  } else {
    print " NOT written in perf";
    $nissues++;
  }
  my $val = $vals{val}{$var}{$indices};
  if (defined $val) {
    print "\t $val";
    if (defined $pval && $pval) {
      my $pctdiff = ($pval - $val) / $val * 100.0;
      print "\t diff = $pctdiff %";
      if (abs($pctdiff) > 5) {
        print " <<<<";
        $nissues++;
      }
    }
  } else {
    print " NOT written in val";
    $nissues++;
  }
  print "\n";
}

print "\n===== Comparisons in perf-write order =====\n".
  "(Will not show missing writes.)\n";
print "Values are from perf, then validation trial\n";
my %nwrites;
for my $pw (@{$writes{perf}}) {
  my ($var, $indices, $pval) = ($pw->[0], $pw->[1], $pw->[2]);
  comp($var, $indices, $pval);
  my $nw = ++$nwrites{$var}{$indices};
  if ($nw > 1) {
    print "^^^^^^^^ $nw writes!!!\n";
    $nissues++;
  }
}

print "\n===== Comparisons in var & index order =====\n";
print "Values are from perf, then validation trial\n";
for my $var (sort keys %pts) {
  for my $indices (sort keys %{$pts{$var}}) {
    comp($var, $indices, $vals{perf}{$var}{$indices});
  }
}

print "\n$0:\n";
for my $key (sort keys %writes) {
  print " ".(scalar @{$writes{$key}})." $key write(s) checked.\n";
}
print " $nissues issue(s) flagged.\n";
print " (Ignore issues outside of local domain when using temporal tiling and MPI.)\n"
  if $nissues;
exit $nissues;
