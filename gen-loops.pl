#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

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

# Purpose: Create loop code.

BEGIN {
    if ($0 =~ m{/}) { ($MYHOME) = $0 =~ m{(.*)/} }
    else { $MYHOME = '.' }
    #print "$MYHOME\n";
}
use lib $MYHOME;
use lib "$MYHOME/lib";
use lib "$MYHOME/../lib";

use 5.010;                      # for smart compare.
use strict;
use warnings;
no warnings qw(portable); # allow 64-bit hex ints.

use File::Basename;
use Text::ParseWords;

# If this module does not exist, download from
# http://search.cpan.org/~stevan/String-Tokenizer-0.05/lib/String/Tokenizer.pm
use String::Tokenizer;

use FileHandle;
use CmdLine;

$| = 1;                         # autoflush.

# Globals.
my %OPT;                        # cmd-line options.
my @dims;                       # names of dimensions.
my @results;                    # names of result buffers.
my $genericCache = "L#";        # a string placeholder for L1 or L2.

# loop-feature bit fields.
my $bSerp = 0x1;                # serpentine path
my $bSquare = 0x2;              # square_wave path
my $bPipe = 0x4;                # pipeline
my $bPrefetch = 0x8;            # prefetch code
my $bSimd = 0x10;               # simd prefix

# Returns next token.
sub checkToken($$$) {
  my $tok = shift;      # token.
  my $allowed = shift;  # regex.
  my $dieIfNotAllowed = shift;

  if (!defined $tok) {
    die "error: unexpected end of input.\n";
  }
  if ($tok !~ /^$allowed$/) {
    if ($dieIfNotAllowed) {
      die "error: illegal token '$tok': expected '$allowed'.\n";
    } else {
      return undef;
    }
  }

  return $tok;
}

# Get next arg (opening paren must already be consumed).
# Return undef if none (closing paren is consumed).
sub getNextArg($$) {
  my $toks = shift;             # ref to token array.
  my $ti = shift;               # ref to token index (starting at paren).

  while (1) {
    my $tok = checkToken($toks->[$$ti++], '\w+|\)|,', 1);

    # comma (ignore).
    if ($tok eq ',') {
    }

    # end.
    elsif ($tok eq ')') {
      return undef;
    }

    # actual token.
    else {
      return $tok;
    }
  }
}

# get a list of simple args.
sub getArgs($$) {
  my $toks = shift;             # ref to token array.
  my $ti = shift;               # ref to token index (starting at paren).

  my @args;
  while (1) {
    my $arg = getNextArg($toks, $ti);
    if ($arg) {
      push @args, $arg;
    } else {
      last;
    }
  }
  return @args;
}

# names of variables based on dimension string(s).

# these must already be defined.
sub beginVar {
    return join('_', 'begin', @_);
}
sub endVar {
    return join('_', 'end', @_);
}
sub stepVar {
    return join('_', 'step', @_);
}

# these are generated.
sub numVar {
    return join('_', 'num', @_);
}
sub indexVar {
    return join('_', 'index', @_);
}
sub loopIndexVar {
    return join('_', 'loop_index', @_);
}
sub startVar {
    return join('_', 'start', @_);
}
sub stopVar {
    return join('_', 'stop', @_);
}
sub pfIndexVar {
    return indexVar(@_)."_pf$genericCache";
}
sub pfLoopIndexVar {
    return loopIndexVar(@_)."_pf$genericCache";
}
sub pfStartVar {
  return startVar(@_)."_pf$genericCache";
}
sub pfStopVar {
  return stopVar(@_)."_pf$genericCache";
}

# this is generated between 0 and numVar when prefetching.
sub midVar {
    return join('_', 'midpoint', @_);
}

# list of start & stop args for all dims.
sub startStopArgs {
    return join(', ', 
                (map { startVar($_) } @dims),
                (map { stopVar($_) } @dims) );
}

# list of start & stop args for all dims for prefetch.
sub pfStartStopArgs {
    my @loopDims = @_;
    my $args = startStopArgs();
    map { my $stVar = startVar($_);
          my $pfStVar = pfStartVar($_);
          $args =~ s/\b$stVar/$pfStVar/g;

          my $spVar = stopVar($_);
          my $pfSpVar = pfStopVar($_);
          $args =~ s/\b$spVar/$pfSpVar/g;
    } @loopDims;
    return $args;
}

# convert strings from/a generic to specific (L1/L2) cache.
sub specifyCache($$) {
    my $strs = shift;           # ref to list.
    my $cache = shift;
    map { s/$genericCache/L$cache/g; } @$strs;
}

# return type of var needed for loop index.
# args: dimension(s).
sub indexType {
    return 'idx_t';
}

# create and init vars *before* beginning of simple or collapsed loop.
sub addIndexVars($$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;           # ref to list of dimensions.

    my $itype = indexType(@$loopDims);

    # number of iterations.
    my @nvars;
    for my $dim (@$loopDims) {
        my $nvar = numVar($dim);
        my $bvar = beginVar($dim);
        my $evar = endVar($dim);
        my $svar = stepVar($dim);
        push @$code, 
        " // Number of iterations to get from $bvar to (but not including) $evar, stepping by $svar.",
        " const idx_t $nvar = (($evar - $bvar) + ($svar - 1)) / $svar;";
        push @nvars, "($itype)$nvar";
    }

    # for collapsed loop, there is not a begin, end, or step for the index.
    if (@$loopDims > 1) {
        my $nvar = numVar(@$loopDims);
        push @$code,
        " // Number of iterations in loop collapsed across ".(join ', ',@$loopDims)." dimensions.",
        " const $itype $nvar = ".join(' * ', @nvars).";";
    }

    # index var.
    push @$code, " // Loop index var.";
    my $ivar = loopIndexVar(@$loopDims);
    push @$code, " $itype $ivar = 0;";
}

# add index variables *inside* the loop.
sub addIndexVars2($$$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;           # ref to list of dimensions.
    my $isPrefetch = shift;     # true if for prefetch vars.
    my $features = shift;           # bits for path types.
    my $loopStack = shift;      # whole stack, including enclosing dims.

    my $itype = indexType(@$loopDims);
    my $civar = loopIndexVar(@$loopDims); # collapsed index var.
    my $pfcivar = pfLoopIndexVar(@$loopDims); # collapsed prefetch index var.
    my $outerDim = $loopDims->[0];
    my $innerDim = $loopDims->[$#$loopDims];

    # prefetch is offset from main index.
    if ($isPrefetch) {
        push @$code, " // Prefetch loop index var.";
        push @$code, " $itype $pfcivar = $civar + PFD$genericCache;";
    }

    # find enclosing dim if avail.
    my $encDim;
    map { $encDim = $loopStack->[$_]
              if $loopStack->[$_ + 1] eq $outerDim; } 0..($#$loopStack-1);
    my $prevDivar;
    $prevDivar = $isPrefetch ? pfIndexVar($encDim) : indexVar($encDim)
        if defined $encDim;
    
    # computed 0-based index var value for each dim.
        my $prevDim = $encDim;
        my $prevNvar;
        my $innerDivar = $isPrefetch ? pfIndexVar($innerDim) : indexVar($innerDim);
        my $innerNvar = numVar($innerDim);

        # loop through each dim, outer to inner.
        for my $i (0..$#$loopDims) {
            my $dim = $loopDims->[$i];
            my $nvar = numVar($dim);
            my $isInner = ($i == $#$loopDims);

            # need to map $ivar to $divar.
            # note that $pfcivar might be >= numVar(@$loopDims).
            my $ivar = $isPrefetch ? $pfcivar : $civar;
            my $divar = $isPrefetch ? pfIndexVar($dim) : indexVar($dim);

            push @$code,
            " // Zero-based, unit-stride ".($isPrefetch ? 'prefetch ' : '')."index var for $dim.";

            # numerator of mod.
            my $num = "$ivar";

            # divisor of index is product of sizes of remaining nested dimensions.
            if (!$isInner) {
                my $div = join('*', map { numVar($loopDims->[$_]) } $i+1 .. $#$loopDims);
                $num .= " / ($div)";
            }

            # mod by size of this dimension (not needed for outer-most dim).
            if ($i > 0) {
                push @$code, " idx_t $divar = ($num) % $nvar;";
            } else {
                push @$code, " idx_t $divar = $num;";
            }

            # apply square-wave to inner 2 dimensions if requested.
            my $isInnerSquare = @$loopDims >=2 && $isInner && ($features & $bSquare);
            if ($isInnerSquare) {

                my $divar2 = "${divar}_x2";
                my $avar = "${prevDivar}_lsb";
                push @$code, 
                " // Modify $prevDivar and $divar for 'square_wave' path.",
                " if (($innerNvar > 1) && ($prevDivar/2 < $prevNvar/2)) {",
                "  // Compute extended $dim index over 2 iterations of $prevDivar.",
                "  idx_t $divar2 = $divar + ($nvar * ($prevDivar & 1));",
                "  // Select $divar from 0,0,1,1,2,2,... sequence",
                "  $divar = $divar2 / 2;",
                "  // Select $prevDivar adjustment value from 0,1,1,0,0,1,1, ... sequence.",
                "  idx_t $avar = ($divar2 & 1) ^ (($divar2 & 2) >> 1);",
                "  // Adjust $prevDivar +/-1 by replacing bit 0.",
                "  $prevDivar = ($prevDivar & (idx_t)-2) | $avar;",
                " } // square-wave.";
            }

            # reverse order of every-other traversal if requested.
            # for inner dim with square-wave, do every 2.
            if (($features & $bSerp) && defined $prevDivar) {
                if ($isInnerSquare) {
                    push @$code,
                    " // Reverse direction of $divar after every-other iteration of $prevDivar for 'square_wave serpentine' path.",
                    " if (($prevDivar & 2) == 2) $divar = $nvar - $divar - 1;";
                } else {
                    push @$code,
                    " // Reverse direction of $divar after every iteration of $prevDivar for  'serpentine' path.",
                    " if (($prevDivar & 1) == 1) $divar = $nvar - $divar - 1;";
                }
            }
            
            $prevDim = $dim;
            $prevDivar = $divar;
            $prevNvar = $nvar;
        }

    # start and stop vars based on individual begin, end, step, and index vars.
    for my $dim (@$loopDims) {
        my $divar = $isPrefetch ? pfIndexVar($dim) : indexVar($dim);
        my $stvar = $isPrefetch ? pfStartVar($dim) : startVar($dim);
        my $spvar = $isPrefetch ? pfStopVar($dim) : stopVar($dim);
        my $bvar = beginVar($dim);
        my $evar = endVar($dim);
        my $svar = stepVar($dim);
        push @$code,
        " // This value of $divar covers $dim from $stvar to $spvar-1.",
        " const idx_t $stvar = $bvar + ($divar * $svar);",
        " const idx_t $spvar = min($stvar + $svar, $evar);";
    }
}

# start simple or collapsed loop body.
sub beginLoop($$$$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;           # ref to list of dimensions.
    my $prefix = shift;         # ref to list of prefix code. May be undef.
    my $endVal = shift;         # value of end-point (start point is already set). May be undef.
    my $features = shift;           # bits for path types.
    my $loopStack = shift;      # whole stack, including enclosing dims.

    $endVal = numVar(@$loopDims) if !defined $endVal;
    my $ivar = loopIndexVar(@$loopDims);
    push @$code, @$prefix if defined $prefix;
    push @$code, " for ( ; $ivar < $endVal; $ivar++) {";

    # add inner index vars.
    addIndexVars2($code, $loopDims, 0, $features, $loopStack);
}

# end simple or collapsed loop body.
sub endLoop($) {
    my $code = shift;           # ref to list of code lines.

    push @$code, " }";
}

# Process the loop-code string.
# This is where most of the work is done.
sub processCode($) {
    my $codeString = shift;

    my $delims = ':;,()[]{}+-*/';
    my $tokenizer = String::Tokenizer->new($codeString, $delims,
                                           String::Tokenizer->IGNORE_WHITESPACE);
    my @toks = $tokenizer->getTokens();
    #print join "\n", @toks;

    # vars to track loops.
    # set at beginning of loop() statements.
    my @loopStack;              # current nesting of dimensions.
    my @loopCounts;             # number of dimensions in each loop.
    my $innerDim;               # iteration dimension of inner loop (undef if not in inner loop).
    my @loopDims;               # dimension(s) of current loop.

    # modifiers before loop() statements.
    my @loopPrefix;             # string(s) to put before loop body.
    my $features = 0;           # bits for loop features.

    # lists of code parts to be output.
    # set at calc() statements.
    my @calcStmts;              # calculation statements.
    my @pfStmtsFullHere;        # full prefetch statement at current start.
    my @pfStmtsFullAhead;       # full prefetch statement at pf start.
    my @pfStmtsEdgeHere;        # edge prefetch statements at current start.
    my @pfStmtsEdgeAhead;       # edge prefetch statements at pf start.

    # final lines of code to output.
    # set at beginning and/or end of loop() statements.
    my @code;

    for (my $ti = 0; $ti <= $#toks; ) {
        my $tok = checkToken($toks[$ti++], '.*', 1);

        # use Intel crew on next loop.
        if (lc $tok eq 'crew') {
            push @loopPrefix, "  // Distribute iterations among HW threads.",
            "CREW_FOR_LOOP";
            warn "info: using Intel crew on following loop.\n";
        }

        # use OpenMP on next loop.
        elsif (lc $tok eq 'omp') {
            my $loopPragma = '_Pragma("omp parallel for ';
            $loopPragma .= "schedule($OPT{ompSchedule})";
            $loopPragma .= '")';
            push @loopPrefix, " // Distribute iterations among OpenMP threads.", 
            $loopPragma;
            warn "info: using OpenMP on following loop.\n";
        }

        # generate prefetch in next loop.
        elsif (lc $tok eq 'prefetch') {

            # turn off compiler prefetch if we are generating prefetch.
            push @loopPrefix, '_Pragma("noprefetch")';
            $features |= $bPrefetch;
            warn "info: generating prefetching in following loop.\n";
        }

        # generate simd in next loop.
        elsif (lc $tok eq 'simd') {

            push @loopPrefix, '_Pragma("simd")';
            $features |= $bSimd;
            warn "info: generating SIMD in following loop.\n";
        }

        # use pipelining in next loop if possible.
        elsif (lc $tok eq 'pipe') {
            $features |= $bPipe;
        }
        
        # use serpentine path in next loop if possible.
        elsif (lc $tok eq 'serpentine') {
            $features |= $bSerp;
        }
        
        # use square_wave path in next loop if possible.
        elsif (lc $tok eq 'square_wave') {
            $features |= $bSquare;
        }
        
        # beginning of a loop.
        # also eats the args in parens and the following '{'.
        elsif (lc $tok eq 'loop') {

            # get loop dimension(s).
            checkToken($toks[$ti++], '\(', 1);
            @loopDims = getArgs(\@toks, \$ti);
            die "error: no args for '$tok'.\n" if @loopDims == 0;
            checkToken($toks[$ti++], '\{', 1);
            push @loopStack, @loopDims;
            push @loopCounts, scalar(@loopDims);

            # set inner dim if applicable.
            undef $innerDim;
            if (scalar(@loopStack) == scalar(@dims)) {
                $innerDim = $loopDims[$#loopDims];

                # check for existence of all vars.
                my @loopVars = sort @loopStack;
                my @dimVars = sort @dims;
                die "error: loop dimensions ".join(', ', @loopStack).
                    " do not match expected dimensions ".join(', ', @dims).".\n"
                    unless @loopVars ~~ @dimVars;
            }

            # check for piping legality.
            if ($features & $bPipe) {
                if (@loopDims == 1) {
                    warn "info: pipelining following loop.\n";
                } else {
                    warn "warning: pipeline requested, but it is not possible because there are ".
                        scalar(@loopDims). " dimensions in following loop.\n";
                    $features &= ~$bPipe;
                }
            }

            # print more info.
            warn "info: collapsing ".scalar(@loopDims). " dimensions in following loop.\n"
                if @loopDims > 1;
            warn "info: generating ".join(', ', @loopDims)." loop...\n";

            # add initial code for index vars, but don't start loop body yet.
            addIndexVars(\@code, \@loopDims);
            
            # if not the inner loop, start the loop body.
            # if it is the inner loop, we might need more than one loop body, so
            # it will be generated when the '}' is seen.
            if (!defined $innerDim) {
                beginLoop(\@code, \@loopDims, \@loopPrefix, undef, $features, \@loopStack);

                # clear data for this loop.
                undef @loopDims;
                undef @loopPrefix;
                $features = 0;
            }
        }

        # thing(s) to calculate.
        # set @*Stmts* vars.
        elsif (lc $tok eq 'calc') {

            die "error: '$tok' attempted outside of inner loop.\n"
                if !defined $innerDim;

            # process things to calculate (args to calc or calc).
            checkToken($toks[$ti++], '\(', 1);
            while (1) {
                my $arg = getNextArg(\@toks, \$ti);
                last if !defined($arg);

                # name of function w/a direction.
                my $fnDir = $arg.'_'.$innerDim; # e.g., fn_z.

                # standard args to functions.
                my $calcArgs = $OPT{comArgs};

                # get optional args from input.
                if (checkToken($toks[$ti], '\(', 0)) {
                    $ti++;
                    my @oargs = getArgs(\@toks, \$ti);
                    $calcArgs .= join(', ', '', @oargs) if (@oargs);
                }
                
                # code for prefetches.
                # e.g., prefetch_L1_fn(...); prefetch_L2_fn_z(...);
                if ($features & $bPrefetch) {
                    push @pfStmtsFullHere, "  $OPT{pfPrefix}${genericCache}_$arg($calcArgs, ".
                        startStopArgs(). ");";
                    push @pfStmtsFullAhead, "  $OPT{pfPrefix}${genericCache}_$arg($calcArgs, ".
                        pfStartStopArgs(@loopDims). ");";
                    push @pfStmtsEdgeHere, "  $OPT{pfPrefix}${genericCache}_$fnDir($calcArgs, ".
                        startStopArgs(). ");";
                    push @pfStmtsEdgeAhead, "  $OPT{pfPrefix}${genericCache}_$fnDir($calcArgs, ".
                        pfStartStopArgs(@loopDims). ");";
                    warn "info: generating prefetch instructions.\n";
                } else {
                    warn "info: not generating prefetch instructions.\n";
                }

                # add pipe prefix and direction suffix to function name.
                # e.g., pipe_fn_z.
                if ($features & $bPipe) {
                    $arg = $OPT{pipePrefix}.$arg.'_'.$innerDim;
                }

                # code for calculations.
                # e.g., calc_fn(...); calc_pipe_fn_z();
                push @calcStmts, "  $OPT{calcPrefix}$arg($calcArgs, ".startStopArgs(). ");";

            }                   # args
        }                       # calc

        # end of loop.
        # this is where most of @code is created.
        elsif ($tok eq '}') {
            die "error: attempt to end loop w/o beginning\n" if !@loopStack;

            # not inner loop?
            # just need to end it.
            if (!defined $innerDim) {

                endLoop(\@code);
            }

            # inner loop.
            # for each part of loop, need to
            # - start it,
            # - add to @code,
            # - end it.
            else {
                
                my $ucDir = uc($innerDim);
                my $pfd = "PFD$genericCache";
                my $iVar = loopIndexVar(@loopDims);
                my $nVar = numVar(@loopDims);

                # declare pipeline vars.
                push @code, " // Pipeline accumulators.", " MAKE_PIPE_$ucDir;"
                    if ($features & $bPipe);

                # prefetch-starting code.
                if ($features & $bPrefetch) {
                    for my $i (0..1) {
                        my $cache = ($i==0) ? 2 : 1; # fetch to L2 first.

                        my @pfCode;
                        push @pfCode, " // Prime prefetch to $genericCache.";
                        
                        # prefetch loop.
                        beginLoop(\@pfCode, \@loopDims, \@loopPrefix, $pfd, $features, \@loopStack);
                        push @pfCode, " // Prefetch to $genericCache.", @pfStmtsFullHere;
                        endLoop(\@pfCode);

                        # convert to specific cache.
                        specifyCache(\@pfCode, $cache);
                        push @code, @pfCode;
                        
                        # reset index to beginning for next loop.
                        push @code, " // Reset index to beginning for next loop.",
                        " $iVar = 0;";
                    }
                }           # PF.

                # pipeline-priming loop.
                if ($features & $bPipe) {

                    # start the loop STENCIL_ORDER before 0.
                    # TODO: make this more general.
                    push @code, " // Prime the calculation pipeline.",
                    " $iVar = -STENCIL_ORDER;";
                    beginLoop(\@code, \@loopDims, \@loopPrefix, 0, $features, \@loopStack);

                    # select only pipe instructions, change calc to prime prefix.
                    my @primeStmts = grep(m=//|$OPT{pipePrefix}=, @calcStmts);
                    map { s/$OPT{calcPrefix}/$OPT{primePrefix}/ } @primeStmts;
                    push @code, @primeStmts;

                    endLoop(\@code);
                }

                # midpoint calculation.
                if ($features & $bPrefetch) {
                    push @code, " // Point where L2-prefetch policy changes.",
                    " // This covers all L1 fetches, even unneeded one(s) beyond end.",
                    " const ".indexType(@loopDims)." ".midVar(@loopDims).
                        " = max($nVar-(PFDL2-PFDL1), $nVar);";
                }

                # computation loop(s):
                # if prefetch:
                # 0: w/L2 prefetch from start to midpoint.
                # 1: w/o L2 prefetch from midpoint to end.
                # if no prefetch:
                # 0: no prefetch from start to end.
                my $lastLoop = ($features & $bPrefetch) ? 1 : 0;
                for my $loop (0 .. $lastLoop) {

                    my $name = "Computation";
                    my $endVal = ($loop == $lastLoop) ?
                        numVar(@loopDims) : midVar(@loopDims);

                    my $comment = " // $name loop.";
                    $comment .= " Same as previous loop, except no L2 prefetch." if $loop==1;
                    push @code, $comment;
                    beginLoop(\@code, \@loopDims, \@loopPrefix, $endVal, $features, \@loopStack);

                    # loop body.
                    push @code, " // $name.", @calcStmts;

                    # prefetch for future iterations.
                    if ($features & $bPrefetch) {
                        for my $i (0..1) {
                            my $cache = ($i==0) ? 2 : 1; # fetch to L2 first.
                            my @pfCode;

                            if ($cache == 2 && $loop == 1) {
                                push @pfCode, " // Not prefetching to $genericCache in this loop.";
                            } else {
                                addIndexVars2(\@pfCode, \@loopDims, 1, $features, \@loopStack);
                                push @pfCode, " // Prefetch to $genericCache.", @pfStmtsFullAhead;
                            }

                            # convert to specific cache.
                            specifyCache(\@pfCode, $cache);
                            push @code, @pfCode;
                        }
                    }       # PF.

                    endLoop(\@code);
                }

                # clear code buffers.
                undef @pfStmtsFullHere;
                undef @pfStmtsFullAhead;
                undef @pfStmtsEdgeHere;
                undef @pfStmtsEdgeAhead;
                undef @calcStmts;

                # clear other data for this loop.
                undef $innerDim;
                undef @loopDims;
                undef @loopPrefix;
                $features = 0;
            }                   # inner loop.

            # pop stacks.
            my $ndims = pop @loopCounts;
            for my $i (1..$ndims) {
                my $sdim = pop @loopStack;
                #push @code, " // End of $sdim loop.";
            }
        }                       # end of a loop.

        # separator (ignore).
        elsif ($tok eq ';') {
        }

        # null or whitespace (ignore).
        elsif ($tok =~ /^\s*$/) {
        }

        else {
            die "error: unrecognized token '$tok'\n";
        }
    }                           # token-handling loop.

    die "error: ".(scalar @loopStack)." loop(s) not closed.\n"
        if @loopStack;

    # indent program avail?
    my $indent = 'indent';
    if (system("which $indent &> /dev/null")) {
        $indent = 'gindent';
        if (system("which $indent &> /dev/null")) {
            warn "note: cannot find an indent utility--output will be unformatted.\n";
            undef $indent;
        }
    }

    # open output stream.
    my $cmd = defined $indent ? "$indent -fca -o $OPT{output} -" :
        "cat > $OPT{output}";
    open OUT, "| $cmd" or die "error: cannot run '$cmd'.\n";

    # header.
    print OUT "/*\n",
    " * ".scalar(@dims)."-D loop code.\n",
    " * Generated automatically from the following pseudo-code:\n *\n";

    # format input.
    my $cmd2 = "echo '$codeString'";
    $cmd2 .= " | $indent -" if (defined $indent);
    open IN, "$cmd2 |" or die "error: cannot run '$cmd2'.\n";
    while (<IN>) {
        print OUT " * $_";
    }
    close IN;
    print OUT " *\n */\n";

    # print out code.
    for my $line (@code) {
        print OUT "\n" if $line =~ m=^\s*//=; # blank lines before comments.
        print OUT "$line\n";
    }

    print OUT "// End of generated code.\n";
    close OUT;
    print "info: output in '$OPT{output}'.\n";
}

# Parse arguments and emit code.
sub main() {

  my(@KNOBS) =
    ( # knob,        description,   optional default
     [ "dims=s", "Comma-separated names of dimensions (in order passed via calls).", 'v,x,y,z'],
     [ "comArgs=s", "Common arguments to all calls (after L1/L2 for prefetch).", 'context'],
     [ "resultBlks=s", "Comma-separated name of block-sized buffers that hold inter-loop values and/or final result at 'save' command.", 'result'],
     [ "calcPrefix=s", "Prefix for calculation call.", 'calc_'],
     [ "primePrefix=s", "Prefix for pipeline-priming call.", 'prime_'],
     [ "pipePrefix=s", "Additional prefix for pipeline call.", 'pipe_'],
     [ "pfPrefix=s", "Prefix for prefetch call.", 'prefetch_'],
     [ "ompSchedule=s", "OpenMP scheduling policy.", "dynamic"],
     [ "output=s", "Name of output file.", 'loops.h'],
    );
  my($command_line) = process_command_line(\%OPT, \@KNOBS);
  print "$command_line\n" if $OPT{verbose};

  my $script = basename($0);
  if (!$command_line || $OPT{help} || @ARGV < 1) {
      print "Outputs C++ code for a loop block.\n",
      "Usage: $script [options] <loop-code-string>\n",
      "Examples:\n",
      "  $script -dims x,y 'loop(x,y) { calc(f); }'\n",
      "  $script -dims x,y,z 'omp loop(x,y) { loop(z) { calc(f); } }'\n",
      "  $script -dims x,y,z 'omp loop(x,y) { prefetch loop(z) { calc(f); } }'\n",
      "  $script -dims x,y,z 'omp loop(x,y) { pipeline loop(z) { calc(f); } }'\n",
      "  $script -dims x,y,z 'omp loop(x) { serpentine loop(y,z) { calc(f); } }'\n",
      "  $script -dims x,y,z 'omp loop(x) { crew loop(y) { loop(z) { calc(f); } } }'\n",
      "Inner loops should contain calc statements that generate calls to calculation functions.\n",
      "A loop statement with more than one argument will generate a single collapsed loop.\n",
      "Optional loop modifiers:\n",
      "  omp:         generate an OpenMP for loop (distribute work across SW threads).\n",
      "  crew:        generate an Intel crew loop (distribute work across HW threads).\n",
      "  prefetch:    generate calls to SW prefetch functions in addition to calculation functions.\n",
      #"  pipe:        generate calls to pipeline versions of calculation functions (deprecated).\n",
      "  serpentine:  generate reverse path when enclosing loop dimension is odd.\n",
      "  square_wave: generate 2D square-wave path for two innermost dimensions of a collapsed loop.\n",
      "For each dim D in dims, loops are generated from begin_D to end_D-1 by step_D;\n",
      "  these vars must be defined outside of the generated code.\n",
      "Each iteration will cover values from start_D to stop_D-1;\n",
      "  these vars will be defined in the generated code.\n",
      "Options:\n";
    print_options_help(\@KNOBS);
    exit 1;
  }

  @dims = split(/\s*,\s*/, $OPT{dims});
  @results = split(/\s*,\s*/, $OPT{resultBlks});

  warn "info: generating ".scalar(@dims)."-D loop code with ".
    scalar(@results)." output(s).\n";

  my $codeString = join(' ', @ARGV);
  processCode($codeString);
}

main();
