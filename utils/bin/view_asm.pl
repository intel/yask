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

# Purpose: expand info in asm files.
# Optionally, find SIMD inner loops in asm file(s) and
# report some stats on them.

use strict;
use File::Basename;

my $minInstrs = 2;
my $printAsm = 0;
my $simdOnly = 0;
my $loopsOnly = 0;
my $targetText = "";
my $targetFn = "";
my @fnames;

sub usage {
  my $msg = shift;

  warn "$msg\n" if defined $msg;
  die "usage: [options] file...\n".
    "options:\n".
    " -p           print instrs in addition to stats\n".
    " -l           view only inner loops\n".
    " -s           view only funcs/loops using SIMD regs\n".
    " -f=<regex>   view only in matching function\n".
    " -t=<regex>   view only if containing matching text\n";
}

for my $arg (@ARGV) {

  if ($arg eq '-h') {
    usage();
  }
  elsif ($arg eq '-p') {
    $printAsm = 1;
  }
  elsif ($arg eq '-l') {
    $loopsOnly = 1;
  }
  elsif ($arg eq '-s') {
    $simdOnly = 1;
  }
  elsif ($arg =~ /^-(\w+)=(.+)$/) {
    my ($key, $val) = ($1, $2);
    if ($key eq "t") { $targetText = $val; }
    elsif ($key eq "f") { $targetFn = $val; }
    else { usage("error: unknown option '$key'"); }
  }
  elsif ($arg =~ /^-/) {
    usage("error: unknown or badly-formatted option '$arg'");
  }
  else {
    die "error: cannot read '$arg'\n" if !-r $arg;
    push @fnames, $arg;
  }
}
usage() if !@fnames;


my @lines;                  # following stats apply to these lines.
my $ninstrs = 0;            # num instrs.
my %istats;                   # instr stats.
my %astats;                   # arg stats.
my %fstats;                   # src-file stats.
my %rstats;                   # SIMD reg stats.
my $curFn = "";

sub clearStats() {
  $ninstrs = 0;
  undef %istats;
  undef %astats;
  undef %fstats;
  undef %rstats;
  undef @lines;
}
    
sub printLines() {
  if ($ninstrs > $minInstrs &&
      (!$simdOnly || scalar %rstats > 0) &&
      (!$targetFn || $curFn =~ /$targetFn/) &&
      (!$targetText || grep(/$targetText/, @lines))) {
    print "\n";
    print "Function '$curFn'\n" if defined $curFn;
    if ($loopsOnly) {
      print "non-" if !scalar %rstats;
      print "SIMD inner loop:\n";
    }
    print @lines if $printAsm;
    print "\n".($loopsOnly ? "Inner loop" : "Function")." summary:\n";
    print "$ninstrs total instrs\n";
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
    print "Num SIMD regs used: ".(scalar keys %rstats)."\n";
  }
}
    

for my $fname (@fnames) {
  my %loopLabels;               # labels that begin inner loops.
  my %files;                    # map from file index to source file-name.
  my %dirs;                     # map from file index to source dir-name.
  my %dirIndices;               # map from dir-name to dir index.

  for my $pass (0..1) {

    my %labels;

    my $asmLine = 0;
    my $getData = 0;
    my ($locInfo, $srcFile); # strings describing current location.

    # Header.
    if (!$pass) {
      print "\n'$fname'...\n";
    } else {
      my %id;
      for my $dir (keys %dirIndices) {
        $id{$dirIndices{$dir}} = $dir;
      }
      print "\nDirectory key:\n";
      for my $di (sort { $a <=> $b } keys %id) {
        print "  <dir$di> = $id{$di}\n";
      }
    }

    open F, "c++filt < $fname |" or usage("error: cannot open '$fname'");
    while (<F>) {
      chomp;

      # file name, e.g.,
      #  .file   4 "myfile.cpp"
      #  .file   13 "foo/bar" "src/stencil_block_loops.hpp"
      if (/^\s*\.file\s+(\d+)\s+"(.*)"/) {
        my ($fi, $fn) = ($1, $2);
        $fn =~ s=" "=/=g;
        $files{$fi} = basename($fn);
        my $dir = dirname($fn);
        $dirs{$fi} = dirname($fn);
        if ($dir && !exists($dirIndices{$dir})) {
          $dirIndices{$dir} = scalar keys %dirIndices;
        }
      }

      # location, e.g.,
      #    .loc    40  23 19 ...
      elsif (/^\s*\.loc\s+(\d+)\s+(\d+)?\s+(\d+)?\s+(.*)/) {
        my ($fi, $line, $col, $info) = ($1, $2, $3, $4);
        if (exists $files{$fi}) {
          $srcFile = $files{$fi};
          $locInfo = "$srcFile";
          $locInfo .= ":$line" if defined $line && $line > 0;
          $locInfo .= ":$col" if defined $col && $col > 0;
          my $srcDir = $dirs{$fi};
          if ($srcDir && exists($dirIndices{$srcDir})) {
            $locInfo = "# <dir$dirIndices{$srcDir}>/$locInfo";
          }
        } else {
          $srcFile = "";
          $locInfo = "";
        }
      }

      # begin function.
      elsif (/#\s+\-+\s+Begin( function)?\s+(.+)/) {
        $curFn = $2;
        #print ">> function $2\n";
        clearStats();
      }

      # end function.
      elsif (/(^\#\s+\-+\s+End)|(\-\- End function)/) {
        printLines() if $pass && !$loopsOnly;
        clearStats();
      }

      # unmangled function name, e.g.,
      # # --- yask::Indices::setFromInitList(yask::Indices *, const std::initializer_list<yask::idx_t> &)
      elsif (/^\#\s+[-]+\s+(\w+)::(.+)/) {
        $curFn = "$1::$2";
      }

      # parameter.
      elsif (/^\#\s+parameter/) {
        push @lines, "$_\n" if !$loopsOnly;
      }

      # label, e.g.,
      #..B1.39:                        # Preds ..B1.54 ..B1.38
      elsif (/^\s*([\w.]+):/) {
        my $lab = $1;
        $labels{$lab} = $asmLine;

        # beginning of an inner loop?
        if ($pass && $loopsOnly && exists $loopLabels{$lab}) {

          # clear previous loop data.
          clearStats();
        }
        push @lines, "$_\n" unless $lab =~ /tmp/;
      }

      # line of asm code, e.g.,
      #   kmovw     %r10d, %k7                                    #137.17
      elsif (/^\s+(\w+)(\s+(.*))?(\#(.*))?/) {
        my ($instr, $args, $comment) = ($1, $3, $5);
        $args = "" if !defined $args;
        $comment = "" if !defined $comment;
        $asmLine++;
        push @lines, "$_\t$locInfo\n";
        $ninstrs++;

        # collect stats in this instr.
        {
          # src stats.
          $fstats{$srcFile}++;

          # instr suffix.
          my $itype = 'non-FP';
          if ($instr =~ /ss/) {
            $itype = 'scalar SP';
          }
          elsif ($instr =~ /sd/) {
            $itype = 'scalar DP';
          }
          elsif ($instr =~ /ps/) {
            $itype = 'packed SP';
          }
          elsif ($instr =~ /pd/) {
            $itype = 'packed DP';
          }

          # SIMD reg stats.
          if ($args =~ /[xyz]mm(\d+)/) {
            $rstats{$1}++;
          }
          
          # arg stats. (dest is last arg.)
          my $type = ($args =~ /[xyz]mm/) ? $& : 'non-SIMD';
          #$type .= ' "spill"' if $comment =~ /spill/i;
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
            $istats{"$itype broadcast"}++;
          } elsif ($instr =~ /vp?align/) {
            $istats{"$itype align"}++;
          } elsif ($instr =~ /vperm\w*2/) {
            $istats{"$itype perm2"}++;
          } elsif ($instr =~ /vperm/) {
            $istats{"$itype perm"}++;
          } elsif ($instr =~ /vfn?m[as]/) {
            $istats{"$itype FMA"}++;
            $istats{"$itype FLOP"} += 2;
          } elsif ($instr =~ /vadd/) {
            $istats{"$itype add"}++;
            $istats{"$itype FLOP"}++;
          } elsif ($instr =~ /vsub/) {
            $istats{"$itype sub"}++;
            $istats{"$itype FLOP"}++;
          } elsif ($instr =~ /vmul/) {
            $istats{"$itype mul"}++;
            $istats{"$itype FLOP"}++;
          } elsif ($instr =~ /vdiv/) {
            $istats{"$itype div"}++;
            $istats{"$itype FLOP"}++;
          } elsif ($instr =~ /vrcp/) {
            $istats{"$itype rcp"}++;
            $istats{"$itype FLOP"}++;
          } elsif ($instr =~ /gather/) {
            $istats{"$itype gather"}++;
          } elsif ($instr =~ /scatter/) {
            $istats{"$itype scatter"}++;
          } elsif ($instr =~ /^vmov/) {
            $istats{"$itype move"}++;
          } elsif ($instr =~ /prefetch/) {
            $istats{"prefetch"}++;
          } elsif ($instr =~ /^v/) {
            $istats{"$itype other"}++;
          } elsif ($instr =~ /^j/) {
            $istats{"jump"}++;
          } elsif ($instr =~ /^cmp/) {
            $istats{"compare"}++;
          } elsif ($instr =~ /^call/) {
            $istats{"call"}++;
          } else {
            $istats{"other instr"}++;
          }
        }

        # jump instr, e.g.,
        #  jb        ..B1.39       # Prob 82%                      #40.2
        if ($instr =~ /^j\w+$/ && $args =~ /^(\S+)/) {
          my $lab = $1;

          # end of a loop?
          # this assumes loops jump backward.
          if (exists $labels{$lab}) {

            # remember for 2nd pass.
            $loopLabels{$lab} = 1;

            # Print loop.
            if ($pass && $loopsOnly) {
              printLines();
            
              # we only want inner loops,
              # so we delete all previous label info now.
              undef %labels;
              
              clearStats();
            }
          }
        }
      }
    }
  }
}

print "To see the asm code for the above, run '$0 -p @ARGV'.\n"
  if !$printAsm;
