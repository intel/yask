#!/usr/bin/env perl

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

# Purpose: find SIMD inner loops in asm file(s) and
# report some stats on them.

use strict;
my $minInstrs = 25;
my $printAsm = 0;

for my $fname (@ARGV) {

  if ($fname eq '-p') {
    $printAsm = 1;
    next;
  }

  my %labels;
  my %loopLabels;
  my %astats;
  my %istats;
  my @lines;

  for my $pass (0..1) {

    my $asmLine = 0;
    my $getData = 0;

    open F, "<$fname" or die "cannot open '$fname'.\n";
    warn "'$fname'...\n" if !$pass;
    while (<F>) {
      chomp;

      # label, e.g.,
      #..B1.39:                        # Preds ..B1.54 ..B1.38
      if (/^\s*(\S+):/) {
        my $lab = $1;
        #warn "$lab: @ $asmLine\n";
        $labels{$lab} = $asmLine;

        # beginning of an inner loop?
        if ($pass && exists $loopLabels{$lab}) {
          undef %istats;
          undef %astats;
          undef @lines;
        }
      }

      # line of code, e.g.,
      #   kmovw     %r10d, %k7                                    #137.17
      elsif (/^\s+(\w+)\s+(.*)\#(.*)/) {
        my ($instr, $args, $comment) = ($1, $2, $3);
        $asmLine++;
        push @lines, "$_\n";

        # jump, e.g.,
        #  jb        ..B1.39       # Prob 82%                      #40.2
        if ($instr =~ /^j\w+$/ && $args =~ /^(\S+)/) {
          my $lab = $1;
          $istats{'jump'}++;

          # end of a loop?
          if (exists $labels{$lab}) {
            my $dist = $asmLine - $labels{$lab};
            $loopLabels{$lab} = 1;
            #warn "$lab: $dist instrs\n";

            # pass 1: print results under certain conditions.
            if ($pass && $dist > $minInstrs && $istats{'SIMD FLOP'}) {
              print "\nSIMD loop $lab:\n".
                "  $dist  instrs\n";
              print @lines if $printAsm;
              print "Instr stats:\n";
              for my $key (sort keys %istats) {
                my $value = $istats{$key};
                printf "%4i  $key\n", $value;
              }
              print "Arg stats:\n";
              for my $key (sort keys %astats) {
                my $value = $astats{$key};
                printf "%4i  $key\n", $value;
              }
            }
          }

          # we only want inner loops,
          # so we can delete all label info now.
          undef %labels;
          undef %istats;
          undef %astats;
        }

        # non-jump instr: collect stats.
        else {

          # arg stats. (dest is last arg.)
          my $type = ($instr =~ /^v/ || $args =~ /[xyz]mm/) ? 'SIMD' : 'non-SIMD';
          $type .= ' "spill"' if $comment =~ /spill/;
          if ($args =~ /\(.*r[bs]p.*\).*,/) {
            $astats{"$type stack load"}++;
          } elsif ($args =~ /,.*\(.*r[bs]p.*\)/) {
            $astats{"$type stack store"}++;
          } elsif ($args =~ /\(.*\).*,/) {
            $astats{"$type non-stack load"}++;
          } elsif ($args =~ /,.*\(.*\)/) {
            $astats{"$type non-stack store"}++;
          }

          # instr stats.
          if ($instr =~ /valign/) {
            $istats{'SIMD valign'}++;
          } elsif ($instr =~ /vperm\w*2/) {
            $istats{'SIMD vperm2'}++;
          } elsif ($instr =~ /vperm/) {
            $istats{'SIMD vperm'}++;
          } elsif ($instr =~ /vfn?m[as]/) {
            $istats{'SIMD FMA'}++;
            $istats{'SIMD FLOP'} += 2;
          } elsif ($instr =~ /vadd/) {
            $istats{'SIMD add'}++;
            $istats{'SIMD FLOP'}++;
          } elsif ($instr =~ /vsub/) {
            $istats{'SIMD sub'}++;
            $istats{'SIMD FLOP'}++;
          } elsif ($instr =~ /vmul/) {
            $istats{'SIMD mul'}++;
            $istats{'SIMD FLOP'}++;
          } elsif ($instr =~ /vdiv/) {
            $istats{'SIMD div'}++;
            $istats{'SIMD FLOP'}++;
          } elsif ($instr =~ /gather/) {
            $istats{'SIMD gather'}++;
          } elsif ($instr =~ /scatter/) {
            $istats{'SIMD scatter'}++;
          } elsif ($instr =~ /^vmov/) {
            $istats{'SIMD move'}++;
          } elsif ($instr =~ /prefetch/) {
            $istats{'prefetch'}++;
          } elsif ($instr =~ /^v/) {
            $istats{'SIMD other'}++;
          } else {
            $istats{'other instr'}++;
          }
        }
      }
    }
  }
}

