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
my $minInstrs = 2;
my $printAsm = 0;
my $targetLabel = "";
my $targetText = "";

sub usage {
  my $msg = shift;

  warn "$msg\n" if defined $msg;
  die "usage: [options] file...\n".
    "options:\n".
    " -p           print instrs\n".
    " -l=<regex>   print only loops at matching label\n".
    " -t=<regex>   print only loops with an matching text\n";
}

for my $arg (@ARGV) {

  if ($arg eq '-h') {
    usage();
  }
  elsif ($arg eq '-p') {
    $printAsm = 1;
    next;
  }
  elsif ($arg =~ /^-(\w+)=(.*)$/) {
    my ($key, $val) = ($1, $2);
    if ($key eq "l") { $targetLabel = $val; }
    elsif ($key eq "t") { $targetText = $val; }
    else { usage("error: unknown option '$key'"); }
    next;
  }

  my $fname = $arg;
  my %files;                    # map from file index to source file-name.
  my %loopLabels;
  my %astats;                   # arg stats.
  my %istats;                   # instr stats.
  my %fstats;                   # src-file stats.

  for my $pass (0..1) {

    my %labels;
    my $asmLine = 0;
    my $getData = 0;
    my ($locInfo, $srcFile);      # strings describing current location.
    my @lines;                  # lines to print.

    open F, "<$fname" or usage("error: cannot open '$fname'");
    print "\n'$fname'...\n" if !$pass;
    while (<F>) {
      chomp;

      # file name, e.g.,
      #  .file   40 "src/stencil_block_loops.hpp"
      if (/^\s*\.file\s+(\d+)\s+"(.*)"/) {
        my ($fi, $fn) = ($1, $2);
        $files{$fi} = $fn;
      }

      # location, e.g.,
      #    .loc    40  23  prologue_end  is_stmt 1
      elsif (/^\s*\.loc\s+(\d+)\s+(.*)/) {
        my ($fi, $info) = ($1, $2);
        if (exists $files{$fi}) {
          $srcFile = $files{$fi};
          $locInfo = "$srcFile:$info";
        } else {
          $srcFile = "";
          $locInfo = "";
        }
      }

      # label, e.g.,
      #..B1.39:                        # Preds ..B1.54 ..B1.38
      elsif (/^\s*(\S+):/) {
        my $lab = $1;
        #warn "$lab: @ $asmLine\n";
        $labels{$lab} = $asmLine;

        # beginning of an inner loop?
        if ($pass && exists $loopLabels{$lab}) {

          # clear loop data.
          undef %istats;
          undef %astats;
          undef %fstats;
          undef @lines;
        }
      }

      # line of code, e.g.,
      #   kmovw     %r10d, %k7                                    #137.17
      elsif (/^\s+(\w+)\s+(.*)\#(.*)/) {
        my ($instr, $args, $comment) = ($1, $2, $3);
        $asmLine++;
        push @lines, "$_\t$locInfo\n";

        # collect stats in this instr.
        {
          # src stats.
          $fstats{$srcFile}++;
          
          # arg stats. (dest is last arg.)
          my $type = ($args =~ /zmm/) ? '512-bit SIMD' :
            ($args =~ /ymm/) ? '128-bit SIMD' :
            ($args =~ /xmm/) ? '64-bit SIMD' :
            ($instr =~ /^v/) ? 'SIMD' : 'non-SIMD';
          #$type .= ' "spill"' if $comment =~ /spill/;
          if ($args =~ /\(.*r[bs]p.*\).*,/) {
            $astats{"$type stack load"}++;
          } elsif ($args =~ /,.*\(.*r[bs]p.*\)/) {
            $astats{"$type stack store"}++;
          } elsif ($args =~ /\(.*\).*,/) {
            $astats{"$type non-stack load"}++;
          } elsif ($args =~ /,.*\(.*\)/) {
            $astats{"$type non-stack store"}++;
          } else {
            $astats{"$type reg only"}++;
          }

          # instr stats.
          if ($instr =~ /vbroadcast/) {
            $istats{'SIMD vbroadcast'}++;
          } elsif ($instr =~ /vp?align/) {
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
          } elsif ($instr =~ /^j/) {
            $istats{'jump'}++;
          } elsif ($instr =~ /^cmp/) {
            $istats{'compare'}++;
          } else {
            $istats{'other instr'}++;
          }
        }

        # jump instr, e.g.,
        #  jb        ..B1.39       # Prob 82%                      #40.2
        if ($instr =~ /^j\w+$/ && $args =~ /^(\S+)/) {
          my $lab = $1;

          # end of a loop?
          # this assumes loops jump backward.
          if (exists $labels{$lab}) {

            my $dist = $asmLine - $labels{$lab};
            $loopLabels{$lab} = 1;
            #warn "$lab: $dist instrs\n";

            # 2nd pass: print results under certain conditions.
            if ($pass &&
                $dist > $minInstrs &&
                $istats{'SIMD FLOP'} &&
                (!$targetLabel || $lab =~ /$targetLabel/) &&
                (!$targetText || grep(/$targetText/, @lines))) {
              print "\nSIMD loop $lab:\n";
              print @lines if $printAsm;
              print "$dist total instrs\n";
              print "Instr counts per instr type (FLOP count is a subtotal):\n";
              for my $key (sort keys %istats) {
                my $value = $istats{$key};
                printf "%4i  $key\n", $value;
              }
              print "Instr counts per operand type:\n";
              for my $key (sort keys %astats) {
                my $value = $astats{$key};
                printf "%4i  $key\n", $value;
              }
              print "Instr counts per source file:\n";
              for my $key (sort keys %fstats) {
                my $value = $fstats{$key};
                printf "%4i  $key\n", $value;
              }
            }

            # we only want inner loops,
            # so we can delete all label info now.
            undef %labels;
            undef %istats;
            undef %astats;
            undef %fstats;
          }
        }
      }
    }
  }
}

print "To see the asm code for the above loop(s), run '$0 -p @ARGV'.\n"
  if !$printAsm;
