#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2017, Intel Corporation
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
no if $] >= 5.017011, warnings => 'experimental::smartmatch';

use File::Basename;
use Text::ParseWords;
use FileHandle;
use CmdLine;

$| = 1;                         # autoflush.

# Globals.
my %OPT;                        # cmd-line options.
my @dims;                       # indices of dimensions.
my $inputVar;                   # input var.
my $cacheLvl = "L#";            # a string placeholder for L1 or L2.

# loop-feature bit fields.
my $bSerp = 0x1;                # serpentine path
my $bSquare = 0x2;              # square_wave path
my $bGroup = 0x4;               # group path
my $bSimd = 0x8;                # simd prefix
my $bPrefetchL1 = 0x10;         # prefetch L1
my $bPrefetchL2 = 0x20;         # prefetch L2
my $bPipe = 0x100;              # pipeline

##########
# Function to make names of variables based on dimension string(s).

# 'idx()' => "".
# 'idx(3)' => "[3]".
# 'idx(3,5)' => "[3][5]".
sub idx {
    return join('', map("[$_]", @_));
}

# inVar() => "$inputVar".
# inVar("foo") => "$inputVar.foo".
# inVar("foo", 5) => "$inputVar.foo[5]".
sub inVar {
    my $vname = shift;
    my $part = (defined $vname) ? ".$vname" : "";
    return "$inputVar$part".idx(@_);
}

# locVar("foo", 5) => "arg_vars.foo[5]".
sub locVar {
    my $vname = shift;
    my $part = (defined $vname) ? ".$vname" : "";
    return "arg_vars$part".idx(@_);
}

# pfVar("foo", 5) => "pfL#_arg_vars.foo[5]".
sub pfVar {
    my $vname = shift;
    my $part = (defined $vname) ? ".$vname" : "";
    return "pf${cacheLvl}_arg_vars$part".idx(@_);
}

# Extract values from input.
sub beginVar {
    return inVar("begin", @_);
}
sub endVar {
    return inVar("end", @_);
}
sub stepVar {
    return inVar("step", @_);
}
sub groupSizeVar {
    return inVar("group_size", @_);
}

# These are generated scalars.
sub numItersVar {
    return join('_', 'num_iters', @_);
}
sub numGroupsVar {
    return join('_', 'num_full_groups', @_);
}
sub numFullGroupItersVar {
    return join('_', 'num_iters_in_full_group', @_);
}
sub numGroupSetItersVar {
    return scalar @_ ? join('_', 'num_iters_in_group_set', @_) :
        'num_iters_in_full_group';
}
sub indexVar {
    return join('_', 'index', @_);
}
sub groupIndexVar {
    return join('_', 'index_of_group', @_);
}
sub groupSetOffsetVar {
    return scalar @_ ? join('_', 'index_offset_within_group_set', @_) :
        'index_offset_within_this_group';
}
sub groupOffsetVar {
    return join('_', 'index_offset_within_this_group', @_);
}
sub numLocalGroupItersVar {
    return join('_', 'num_iters_in_group', @_);
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
    return indexVar(@_)."_pf$cacheLvl";
}
sub pfLoopIndexVar {
    return loopIndexVar(@_)."_pf$cacheLvl";
}
sub pfStartVar {
  return startVar(@_)."_pf$cacheLvl";
}
sub pfStopVar {
  return stopVar(@_)."_pf$cacheLvl";
}

# this is generated between 0 and numItersVar when prefetching.
sub midVar {
    return join('_', 'midpoint', @_);
}

# return string of all non-empty args separated by commas.
sub joinArgs {
    return join(', ', grep(/./, @_));
}

# dimension comment string.
sub dimStr {
    return '0 dimensions' if @_ == 0;
    my $s = "dimension";
    $s .= 's' if @_ > 1;
    $s .= ' '.joinArgs(@_);
    return $s;
}

# make args for a call.
sub makeArgs {
    my @loopDims = @_;

    my @stmts;
    push @stmts, " ScanIndices ".locVar()."(".inVar().");";
    map {
        push @stmts,
            " ".locVar("start", $_)." = ".startVar($_).";",
            " ".locVar("stop", $_)." = ".stopVar($_).";",
            " ".locVar("index", $_)." = ".indexVar($_).";",
    } @loopDims;
    return @stmts;
}

# make args for a prefetch call.
sub makePfArgs {
    my @loopDims = @_;

    my @stmts;
    push @stmts, " ScanIndices ".pfVar()."(".locVar().");";
    map {
        push @stmts,
            " ".pfVar("start", $_)." = ".pfStartVar($_).";",
            " ".pfVar("stop", $_)." = ".pfStopVar($_).";",
            " ".pfVar("index", $_)." = ".pfIndexVar($_).";",
    } @loopDims;
    return @stmts;
}

# convert strings from/a generic to specific (L1/L2) cache.
sub specifyCache($$) {
    my $strs = shift;           # ref to list.
    my $cache = shift;
    map { s/$cacheLvl/L$cache/g; } @$strs;
}

###########
# Loop-constructing functions.

# return type of var needed for loop index.
# args: dimension(s) -- currently ignored.
sub indexType {
    return 'idx_t';
}

# Create and init vars *before* beginning of simple or collapsed loop.
sub addIndexVars($$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;           # ref to list of dimensions.
    my $features = shift;       # bits for path types.

    push @$code,
        " // ** Begin scan over ".dimStr(@$loopDims).". **";
    
    my $itype = indexType(@$loopDims);

    for my $pass (0..1) {
        for my $i (0..$#$loopDims) {
            my $dim = $loopDims->[$i];
            my $isInner = ($i == $#$loopDims);

            # Pass 0: iterations.
            if ($pass == 0) {
                my $nvar = numItersVar($dim);
                my $bvar = beginVar($dim);
                my $evar = endVar($dim);
                my $svar = stepVar($dim);
                my $ntvar = numGroupsVar($dim);
                my $tsvar = groupSizeVar($dim);
                my $ntivar = numFullGroupItersVar($dim);

                push @$code, 
                " // Number of iterations to get from $bvar to (but not including) $evar, stepping by $svar.".
                " This value is rounded up because the last iteration may cover fewer than $svar steps.",
                " const $itype $nvar = (($evar - $bvar) + ($svar - 1)) / $svar;";

                # For grouped loops.
                if ($features & $bGroup) {

                    # loop iterations within one group.
                    push @$code,
                    " // Number of iterations in one full group in $dim dimension.".
                    " This value is rounded up, effectively increasing the group size if needed".
                        " to a multiple of $svar.",
                    " const $itype $ntivar = std::min(($tsvar + ($svar - 1)) / $svar, $nvar);";

                    # number of full groups.
                    push @$code, 
                    " // Number of *full* groups in $dim dimension.",
                    " const $itype $ntvar = $nvar / $ntivar;";
                }
            }

            # Pass 1: Product of sizes of this and remaining nested dimensions.
            elsif (!$isInner) {
                my @subDims = @$loopDims[$i .. $#$loopDims];
                my $loopStr = dimStr(@subDims);

                # Product of iterations.
                my $snvar = numItersVar(@subDims);
                my $snval = join(' * ', map { numItersVar($_) } @subDims);
                push @$code,
                " // Number of iterations in $loopStr",
                " const $itype $snvar = $snval;";
            }
        }
    }
}

# Add index variables *inside* the loop.
# TODO: add prefetch for grouping.
sub addIndexVars2($$$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions in loop.
    my $isPrefetch = shift;     # true if for prefetch vars.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack, including enclosing dims.

    my $itype = indexType(@$loopDims);
    my $civar = loopIndexVar(@$loopDims);     # collapsed index var; everything based on this.
    my $pfcivar = pfLoopIndexVar(@$loopDims); # collapsed prefetch index var.
    my $outerDim = $loopDims->[0];            # outer dim of these loops.
    my $innerDim = $loopDims->[$#$loopDims];  # inner dim of these loops.

    # Grouping.
    if ($features & $bGroup) {

        die "error: prefetching not compatible with grouping.\n"
            if $isPrefetch;
        die "error: serpentine not compatible with grouping.\n"
            if $features & $bSerp;
        die "error: square-wave not compatible with grouping.\n"
            if $features & $bSquare;

        my $ndims = scalar @$loopDims;

        # declare local size vars.
        push @$code, " // Working vars for iterations in groups.".
            " These are initialized to full-group counts and then".
            " reduced if we are in a partial group.";
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $ltvar = numLocalGroupItersVar($dim);
            my $ltval = numFullGroupItersVar($dim);
            push @$code, " $itype $ltvar = $ltval;";
        }

        # calculate group indices and sizes and 1D offsets within groups.
        my $prevOvar = $civar;  # previous offset.
        for my $i (0 .. $ndims-1) {

            # dim at $i.
            my $dim = $loopDims->[$i];

            # dims up to (outside of) $i (empty for outer dim)
            my @outDims = @$loopDims[0 .. $i - 1];
            
            # dims up to (outside of) and including $i.
            my @dims = @$loopDims[0 .. $i];
            
            # dims after (inside of) $i (empty for inner dim)
            my @inDims = @$loopDims[$i + 1 .. $ndims - 1];
            my $inStr = dimStr(@inDims);

            # Size of group set.
            my $tgvar = numGroupSetItersVar(@inDims);
            my $tgval = join(' * ', 
                             (map { numLocalGroupItersVar($_) } @dims),
                             (map { numItersVar($_) } @inDims));
            my $tgStr = @inDims ?
                "the set of groups across $inStr" : "this group";
            push @$code,
            " // Number of iterations in $tgStr.",
            " $itype $tgvar = $tgval;";

            # Index of this group in this dim.
            my $tivar = groupIndexVar($dim);
            my $tival = "$prevOvar / $tgvar";
            push @$code,
            " // Index of this group in $dim dimension.",
            " $itype $tivar = $tival;";
            
            # 1D offset within group set.
            my $ovar = groupSetOffsetVar(@inDims);
            my $oval = "$prevOvar % $tgvar";
            push @$code,
            " // Linear offset within $tgStr.",
            " $itype $ovar = $oval;";
            
            # Size of this group in this dim.
            my $ltvar = numLocalGroupItersVar($dim);
            my $ltval = numItersVar($dim).
                " - (".numGroupsVar($dim)." * ".numFullGroupItersVar($dim).")";
            push @$code,
            " // Adjust number of iterations in this group in $dim dimension.",
            " if ($tivar >= ".numGroupsVar($dim).")".
                "  $ltvar = $ltval;";

            # for next dim.
            $prevOvar = $ovar;
        }

        # Calculate nD indices within group and overall.
        # TODO: allow different paths *within* group.
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $tivar = groupIndexVar($dim);
            my $ovar = groupSetOffsetVar(); # last one calculated above.

            # dims after (inside of) $i (empty for inner dim)
            my @inDims = @$loopDims[$i + 1 .. $ndims - 1];
            
            # Determine offset within this group.
            my $dovar = groupOffsetVar($dim);
            my $doval = $ovar;

            # divisor of index is product of sizes of remaining nested dimensions.
            if (@inDims) {
                my $subVal = join(' * ', map { numLocalGroupItersVar($_) } @inDims);
                $doval .= " / ($subVal)";
            }

            # mod by size of this dimension (not needed for outer-most dim).
            if ($i > 0) {
                $doval = "($doval) % ".numLocalGroupItersVar($dim);
            }

            # output offset in this dim.
            push @$code,
            " // Offset within this group in $dim dimension.",
            " $itype $dovar = $doval;";

            # final index in this dim.
            my $divar = indexVar($dim);
            my $dival = numFullGroupItersVar($dim)." * $tivar + $dovar";
            push @$code,
            " // Zero-based, unit-stride index for ".dimStr($dim).".",
            " $itype $divar = $dival;";
        }
    }

    # No grouping.
    else {

        # prefetch is offset from main index.
        if ($isPrefetch) {
            push @$code, " // Prefetch loop index var.",
            " $itype $pfcivar = $civar + PFD$cacheLvl;";
        }

        # find enclosing dim outside of these loops if avail.
        my $encDim;
        map { $encDim = $loopStack->[$_]
                  if $loopStack->[$_ + 1] eq $outerDim; } 0..($#$loopStack-1);
        my $prevDivar;
        $prevDivar = indexVar($encDim)
            if defined $encDim;

        # computed 0-based index var value for each dim.
        my $prevDim = $encDim;
        my $prevNvar;
        my $innerDivar = $isPrefetch ? pfIndexVar($innerDim) : indexVar($innerDim);
        my $innerNvar = numItersVar($innerDim);

        # loop through each dim, outer to inner.
        for my $i (0..$#$loopDims) {
            my $dim = $loopDims->[$i];
            my $nvar = numItersVar($dim);
            my $isInner = ($i == $#$loopDims);

            # Goal is to compute $divar from 1D $ivar.
            # note that $pfcivar might be >= numItersVar(@$loopDims).
            my $ivar = $isPrefetch ? $pfcivar : $civar;
            my $divar = $isPrefetch ? pfIndexVar($dim) : indexVar($dim);

            # Determine $divar value: actual index in this dimension.
            my $dival = $ivar;

            # divisor of index is product of sizes of remaining nested dimensions.
            if (!$isInner) {
                my @subDims = @$loopDims[$i+1 .. $#$loopDims];
                my $snvar = numItersVar(@subDims);
                $dival .= " / $snvar";
            }

            # mod by size of this dimension (not needed for outer-most dim).
            if ($i > 0) {
                $dival = "($dival) % $nvar";
            }

            # output $divar.
            push @$code,
            " // Zero-based, unit-stride ".($isPrefetch ? 'prefetch ' : '').
                "index for ".dimStr($dim).".",
            " idx_t $divar = $dival;";

            # apply square-wave to inner 2 dimensions if requested.
            my $isInnerSquare = @$loopDims >=2 && $isInner && ($features & $bSquare);
            if ($isInnerSquare) {

                my $divar2 = "index_x2";
                my $avar = "lsb";
                push @$code, 
                " // Modify $prevDivar and $divar for 'square_wave' path.",
                " if (($innerNvar > 1) && ($prevDivar/2 < $prevNvar/2)) {",
                "  // Compute extended index over 2 iterations of $prevDivar.",
                "  idx_t $divar2 = $divar + ($nvar * ($prevDivar & 1));",
                "  // Select $divar from 0,0,1,1,2,2,... sequence",
                "  $divar = $divar2 / 2;",
                "  // Select $prevDivar adjustment value from 0,1,1,0,0,1,1, ... sequence.",
                "  idx_t $avar = ($divar2 & 0x1) ^ (($divar2 & 0x2) >> 1);",
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
        " // This value of $divar covers ".dimStr($dim)." from $stvar to $spvar-1.",
        " idx_t $stvar = $bvar + ($divar * $svar);",
        " idx_t $spvar = std::min($stvar + $svar, $evar);";
    }
}

# start simple or collapsed loop body.
sub beginLoop($$$$$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions.
    my $prefix = shift;         # ref to list of prefix code. May be undef.
    my $beginVal = shift;       # beginning of loop.
    my $endVal = shift;         # end of loop (undef to use default).
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack, including enclosing dims.

    $endVal = numItersVar(@$loopDims) if !defined $endVal;
    my $itype = indexType(@$loopDims);
    my $ivar = loopIndexVar(@$loopDims);
    push @$code, @$prefix if defined $prefix;
    push @$code, " for ($itype $ivar = $beginVal; $ivar < $endVal; $ivar++) {";

    # add inner index vars.
    addIndexVars2($code, $loopDims, 0, $features, $loopStack);
}

# end simple or collapsed loop body.
sub endLoop($) {
    my $code = shift;           # ref to list of code lines.

    push @$code, " }";
}

##########
# Parsing functions.

# Split a string into tokens, ignoring whitespace.
sub tokenize($) {
    my $str = shift;
    my @toks;

    while (length($str)) {

        # default is 1 char.
        my $len = 1;
        
        # A series of chars and/or digits.
        if ($str =~ /^\w+/) {
            $len = length($&);
        }
            
        # A series of 2 or more dots.
        elsif ($str =~ /^\.\.+/) {
            $len = length($&);
        }
            
        # get a token.
        my $tok = substr($str, 0, $len, '');

        # keep unless WS.
        push @toks, $tok unless $tok =~ /^\s$/;
    }
    return @toks;
}

# Returns next token if match to allowed.
# If not match, return undef or die.
sub checkToken($$$) {
  my $tok = shift;      # token to look at.
  my $allowed = shift;  # regex to match.
  my $dieIfNotAllowed = shift;

  # die if at end.
  if (!defined $tok) {
    die "error: unexpected end of input.\n";
  }

  # check match.
  if ($tok !~ /^$allowed$/) {
    if ($dieIfNotAllowed) {
      die "error: illegal token '$tok': expected '$allowed'.\n";
    } else {
      return undef;
    }
  }

  return $tok;
}

# Determine whether we are in the inner loop.
sub isInInner($$) {
  my $toks = shift;             # ref to token array.
  my $ti = shift;               # ref to token index.

  # Scan for next brace.
  for (my $i = $$ti; $i <= $#$toks; $i++) {

      my $tok = $toks->[$i];
      if ($tok eq '{') {
          return 0;             # starting another loop.
      }
      elsif ($tok eq '}') {
          return 1;             # end of loop.
      }
  }
  return 0;                     # should not get here.
}

# Get next arg (opening paren must already be consumed).
# Return undef if none (closing paren is consumed).
sub getNextArg($$) {
  my $toks = shift;             # ref to token array.
  my $ti = shift;               # ref to token index (starting at paren).

  my $N = scalar(@dims);
  while (1) {
    my $tok = checkToken($toks->[$$ti++], '\w+|N[-+]|\,|\.+|\)', 1);

    # comma (ignore).
    if ($tok eq ',') {
    }

    # end (done).
    elsif ($tok eq ')') {
      return undef;
    }

    # actual token.
    else {

        # Handle, e.g., 'N+1', 'N-2'.
        if ($tok eq 'N') {
            my $oper = checkToken($toks->[$$ti++], '[-+]', 1);
            my $tok2 = checkToken($toks->[$$ti++], '\d+', 1);
            if ($oper eq '+') {
                $tok = $N + $tok2;
            } else {
                $tok = $N - $tok2;
            }
        }

        return $tok;
    }
  }
}

# get a list of args until the next ')'.
sub getArgs($$) {
    my $toks = shift;           # ref to token array.
    my $ti = shift;             # ref to token index (starting at paren).

    my $prevArg;
    my @args;
    while (1) {
        my $arg = getNextArg($toks, $ti);

        # end.
        if (!defined $arg) {
            last;
        }

        # Handle '..'.
        elsif ($arg =~ /^\.+$/) {
            die "Error: missing token before '$arg'.\n"
                if !defined $prevArg;
            die "Error: non-numerical token before '$arg'.\n"
                if $prevArg !~ /^\d+$/;
            my $arg2 = getNextArg($toks, $ti);
            die "Error: missing token after '$arg'.\n"
                if !defined $arg2;
            die "Error: non-numerical token after '$arg'.\n"
                if $arg2 !~ /^\d+$/;
            for my $i ($prevArg+1 .. $arg2) {
                push @args, $i;
            }
        }

        else {
            push @args, $arg;
            $prevArg = $arg;
        }
    }
    return @args;
}

# Process the loop-code string.
# This is where most of the work is done.
sub processCode($) {
    my $codeString = shift;

    my @toks = tokenize($codeString);
    ##print join "\n", @toks;

    # vars to track loops.
    # set at beginning of loop() statements.
    my @loopStack;              # current nesting of dimensions.
    my @loopCounts;             # number of dimensions in each loop.
    my $curInnerDim;            # iteration dimension of inner loop (undef if not in inner loop).
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
            my $loopPragma = "_Pragma(\"$OPT{ompConstruct}\")";
            push @loopPrefix, " // Distribute iterations among OpenMP threads.", 
            $loopPragma;
            warn "info: using OpenMP on following loop.\n";
        }

        # generate prefetch in next loop.
        elsif (lc $tok eq 'prefetch') {

            # get optional args from input.
            my @pfargs;
            if (checkToken($toks[$ti], '\(', 0)) {
                $ti++;
                @pfargs = getArgs(\@toks, \$ti);
                for my $i (0..$#pfargs) {
                    if ($pfargs[$i] =~ /^l([12])$/i) {
                        $features |= ($1 == 1) ? $bPrefetchL1 : $bPrefetchL2;
                    } else {
                        die "error: argument to 'prefetch' must be 'L1' or 'L2'\n";
                    }
                }
            }

            # if no args, prefetch L1 and L2.
            if (@pfargs == 0) {
                $features |= $bPrefetchL1;
                $features |= $bPrefetchL2;
            }
            
            # turn off compiler prefetch if we are generating prefetch.
            push @loopPrefix, '_Pragma("noprefetch")';
            warn "info: generating prefetching in following loop.\n";
        }

        # generate simd in next loop.
        elsif (lc $tok eq 'simd') {

            push @loopPrefix, '_Pragma("simd")';
            $features |= $bSimd;
            warn "info: generating SIMD in following loop.\n";
        }

        # use pipelining in next loop if possible.
        elsif (lc $tok eq 'pipeline') {
            $features |= $bPipe;
        }
        
        # use grouped path in next loop if possible.
        elsif (lc $tok eq 'grouped') {
            $features |= $bGroup;
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
            checkToken($toks[$ti++], '\{', 1); # eat the '{'.
            push @loopStack, @loopDims;
            push @loopCounts, scalar(@loopDims);

            # check for existence of all vars.
            for my $ld (@loopDims) {
                die "Error: loop variable '$ld' not in ".dimStr(@dims).".\n"
                    if !grep($_ eq $ld, @dims);
            }
            
            # set inner dim if applicable.
            undef $curInnerDim;
            if (isInInner(\@toks, \$ti)) {
                $curInnerDim = $loopDims[$#loopDims];
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

            # TODO: check for conflicting features like omp with prefetch.

            # print more info.
            warn "info: generating scan over ".dimStr(@loopDims)."...\n";

            # add initial code for index vars, but don't start loop body yet.
            addIndexVars(\@code, \@loopDims, $features);
            
            # if not the inner loop, start the loop body.
            # if it is the inner loop, we might need more than one loop body, so
            # it will be generated when the '}' is seen.
            if (!defined $curInnerDim) {
                beginLoop(\@code, \@loopDims, \@loopPrefix, 0, undef, $features, \@loopStack);

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
                if !defined $curInnerDim;

            # process things to calculate (args to calc).
            checkToken($toks[$ti++], '\(', 1);
            my $ncalc = 0;
            while (1) {
                my $arg = getNextArg(\@toks, \$ti);
                last if !defined($arg);
                $ncalc++;

                # Edge suffix for simple (non-collapsed) loops.
                my $edgeSuf = '';
                $edgeSuf = '_dir_'.$curInnerDim if @loopDims == 1;

                # standard args to functions.
                my $calcArgs = $OPT{comArgs};

                # get optional args from input.
                if (checkToken($toks[$ti], '\(', 0)) {
                    $ti++;
                    my @oargs = getArgs(\@toks, \$ti);
                    $calcArgs = joinArgs($calcArgs, @oargs) if (@oargs);
                }
                
                # generic code for prefetches.
                # e.g., prefetch_fn<L#>(...); prefetch_fn_dir_3<L#>(...);
                if ($features & ($bPrefetchL1 | $bPrefetchL2)) {
                    if ($ncalc == 1) {
                        my @pfArgs = makeArgs(@loopStack);
                        push @pfStmtsFullHere, @pfArgs;
                        push @pfStmtsEdgeHere, @pfArgs;
                        @pfArgs = makePfArgs(@loopDims);
                        push @pfStmtsFullAhead, @pfArgs;
                        push @pfStmtsEdgeAhead, @pfArgs;
                    }
                    push @pfStmtsFullHere, 
                        "  $OPT{pfPrefix}$arg<$cacheLvl>(".
                        joinArgs($calcArgs, locVar()). ");";
                    push @pfStmtsEdgeHere, 
                        "  $OPT{pfPrefix}$arg$edgeSuf<$cacheLvl>(".
                        joinArgs($calcArgs, locVar). ");";
                    push @pfStmtsFullAhead,
                        "  $OPT{pfPrefix}$arg<$cacheLvl>(".
                        joinArgs($calcArgs, pfVar()). ");";
                    push @pfStmtsEdgeAhead,
                        "  $OPT{pfPrefix}$arg$edgeSuf<$cacheLvl>(".
                        joinArgs($calcArgs, pfVar()). ");";
                    warn "info: generating prefetch instructions.\n";
                } else {
                    warn "info: not generating prefetch instructions.\n";
                }

                # add pipe prefix and direction suffix to function name.
                # e.g., pipe_fn_z.
                if ($features & $bPipe) {
                    $arg = $OPT{pipePrefix}.$arg.'_'.$curInnerDim;
                }

                # code for calculations.
                # e.g., calc_fn(...); calc_pipe_fn_z();
                push @calcStmts, makeArgs(@loopStack)
                    if $ncalc == 1;
                push @calcStmts,
                    "  $OPT{calcPrefix}$arg(".
                    joinArgs($calcArgs, locVar()). ");";

            }                   # args
        }                       # calc

        # end of loop.
        # this is where most of @code is created.
        elsif ($tok eq '}') {
            die "error: attempt to end loop w/o beginning\n" if !@loopStack;

            # not inner loop?
            # just need to end it.
            if (!defined $curInnerDim) {

                endLoop(\@code);
            }

            # inner loop.
            # for each part of loop, need to
            # - start it,
            # - add to @code,
            # - end it.
            else {
                
                my $ucDir = uc($curInnerDim);
                my $pfd = "PFD$cacheLvl";
                my $nVar = numItersVar(@loopDims);
                my $doSplitL2 = ($features & $bPrefetchL2) && $OPT{splitL2};

                # declare pipeline vars.
                push @code, " // Pipeline accumulators.", " MAKE_PIPE_$ucDir;"
                    if ($features & $bPipe);

                # check prefetch settings.
                if (($features & $bPrefetchL1) && ($features & $bPrefetchL2)) {
                    push @code, " // Check prefetch settings.",
                    "#if PFDL2 <= PFDL1",
                    '#error "PFDL2 <= PFDL1"',
                    "#endif";
                }
                
                # prefetch-starting loop(s).
                # TODO: generate full prefetch once, then edge ones.
                for my $i (0..1) {
                    my $cache = ($i==0) ? 2 : 1; # fetch to L2 first.
                    if (($cache == 1 && ($features & $bPrefetchL1)) ||
                        ($cache == 2 && ($features & $bPrefetchL2))) {

                        my @pfCode;
                        push @pfCode, " // Prime prefetch to $cacheLvl.";
                        
                        # prefetch loop.
                        beginLoop(\@pfCode, \@loopDims, \@loopPrefix, 0, $pfd, $features, \@loopStack);
                        push @pfCode, " // Prefetch to $cacheLvl.", @pfStmtsFullHere;
                        endLoop(\@pfCode);
                        
                        # convert to specific cache.
                        specifyCache(\@pfCode, $cache);
                        push @code, @pfCode;
                    }
                }           # PF.

                # pipeline-priming loop.
                if ($features & $bPipe) {

                    # start the loop STENCIL_ORDER before 0.
                    # TODO: make this more general.
                    push @code, " // Prime the calculation pipeline.";
                    beginLoop(\@code, \@loopDims, \@loopPrefix, "-STENCIL_ORDER", 0, $features, \@loopStack);

                    # select only pipe instructions, change calc to prime prefix.
                    my @primeStmts = grep(m=//|$OPT{pipePrefix}=, @calcStmts);
                    map { s/$OPT{calcPrefix}/$OPT{primePrefix}/ } @primeStmts;
                    push @code, @primeStmts;

                    endLoop(\@code);
                }

                # midpoint calculation for L2 prefetch only.
                if ($doSplitL2) {
                    my $ofs = ($features & $bPrefetchL1) ? "(PFDL2-PFDL1)" : "PFDL2";
                    push @code, " // Point where L2-prefetch policy changes.";
                    push @code, " // This covers all L1 fetches, even unneeded one(s) beyond end."
                        if ($features & $bPrefetchL1);
                    push @code, " const ".indexType(@loopDims)." ".midVar(@loopDims).
                        " = std::max($nVar-$ofs, $nVar);";
                }

                # 1 or 2 computation loop(s):
                # if L2 prefetch:
                #  loop 0: w/L2 prefetch from start to midpoint.
                #  loop 1: w/o L2 prefetch from midpoint to end.
                # if no L2 prefetch:
                #  loop 0: no L2 prefetch from start to end.
                my $lastLoop = $doSplitL2 ? 1 : 0;
                for my $loop (0 .. $lastLoop) {

                    my $name = "Computation";
                    my $endVal = ($loop == $lastLoop) ?
                        numItersVar(@loopDims) : midVar(@loopDims);
                    my $beginVal = ($lastLoop > 0 && $loop == $lastLoop) ?
                        midVar(@loopDims) : 0;

                    my $comment = " // $name loop.";
                    $comment .= " Same as previous loop, except no L2 prefetch." if $loop==1;
                    push @code, $comment;
                    push @code, $OPT{innerMod};
                    beginLoop(\@code, \@loopDims, \@loopPrefix, 
                              $beginVal, $endVal, $features, \@loopStack);

                    # loop body.
                    push @code, " // $name.", @calcStmts;

                    # prefetch for future iterations.
                    for my $i (0..1) {
                        my $cache = ($i==0) ? 2 : 1; # fetch to L2 first.
                        if (($cache == 1 && ($features & $bPrefetchL1)) ||
                            ($cache == 2 && ($features & $bPrefetchL2))) {
                            
                            my @pfCode;

                            if ($cache == 2 && $loop == 1) {
                                push @pfCode, " // Not prefetching to $cacheLvl in this loop.";
                            } else {
                                addIndexVars2(\@pfCode, \@loopDims, 1, $features, \@loopStack);
                                push @pfCode, " // Prefetch to $cacheLvl.", @pfStmtsEdgeAhead;
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
                undef $curInnerDim;
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
        " * ".scalar(@dims)."-D grid-scanning code.\n",
        " * Generated automatically from the following pseudo-code:\n",
        " *\n",
        " * N = ",$#dims,";\n";

    # format input.
    my $cmd2 = "echo '$codeString'";
    $cmd2 .= " | $indent -" if (defined $indent);
    open IN, "$cmd2 |" or die "error: cannot run '$cmd2'.\n";
    while (<IN>) {
        print OUT " * $_";
    }
    close IN;
    print OUT " *\n */\n\n";

    # print out code.
    print OUT "// 'ScanIndices $inputVar' must be set before the following code.\n",
        "{\n";
    for my $line (@code) {
        print OUT "\n" if $line =~ m=^\s*//=; # blank line before comment.
        print OUT " $line\n";
    }
    print OUT "}\n",
        "// End of generated code.\n";
    close OUT;
    print "info: output in '$OPT{output}'.\n";
}

# Parse arguments and emit code.
sub main() {

    my(@KNOBS) = (
        # knob,        description,   optional default
        [ "ndims=i", "Value of N.", 1],
        [ "inVar=s", "Input index vars.", 'scanVars'],
        [ "comArgs=s", "Common arguments to all calls (after L1/L2 for prefetch).", ''],
        [ "calcPrefix=s", "Prefix for calculation call.", 'calc_'],
        [ "pfPrefix=s", "Prefix for prefetch call.", 'prefetch_'],
        #[ "primePrefix=s", "Prefix for pipeline-priming call.", 'prime_'],
        #[ "pipePrefix=s", "Additional prefix for pipeline call.", 'pipe_'],
        [ "ompConstruct=s", "Pragma to use before 'omp' loop(s).", "omp parallel for"],
        [ "innerMod=s", "Code to insert before inner computation loops.",
          '_Pragma("nounroll_and_jam") _Pragma("nofusion")'],
        [ "splitL2!", "Split inner loops with/without L2 prefetching.", 0],
        [ "output=s", "Name of output file.", 'loops.h'],
        );
    my($command_line) = process_command_line(\%OPT, \@KNOBS);
    print "$command_line\n" if $OPT{verbose};

    my $script = basename($0);
    if (!$command_line || $OPT{help} || @ARGV < 1) {
        print "Outputs C++ code to scan N-D grids.\n",
            "Usage: $script [options] <code-string>\n",
            "The <code-string> contains optionally-nested scans across the given",
            "  indices between 0 and N-1 indicated by 'loop(<indices>)'\n",
            "Indices may be specified as a comma-separated list or <first..last> range,\n",
            "  using the variable 'N' as needed.\n",
            "Inner loops should contain calc statements that generate calls to calculation functions.\n",
            "A loop statement with more than one argument will generate a single collapsed loop.\n",
            "Optional loop modifiers:\n",
            "  omp:             generate an OpenMP for loop (distribute work across SW threads).\n",
            #"  crew:            generate an Intel crew loop (distribute work across HW threads).\n",
            "  prefetch:        generate calls to SW L1 & L2 prefetch functions in addition to calc functions.\n",
            "  prefetch(L1,L2): generate calls to SW L1 & L2 prefetch functions in addition to calc functions.\n",
            "  prefetch(L1):    generate calls to SW L1 prefetch functions in addition to calc functions.\n",
            "  prefetch(L2):    generate calls to SW L2 prefetch functions in addition to calc functions.\n",
            "  grouped:         generate grouped path within a collapsed loop.\n",
            "  serpentine:      generate reverse path when enclosing loop dimension is odd.\n",
            "  square_wave:     generate 2D square-wave path for two innermost dimensions of a collapsed loop.\n",
            #"  pipeline:        generate calls to pipeline versions of calculation functions (deprecated).\n",
            "A 'ScanIndices' var must be defined in C++ code prior to including the generated code.\n",
            "  This struct contains the following 'Indices' elements:\n",
            "  'begin':       [in] first index to scan in each dim.\n",
            "  'end':         [in] one past last index to scan in each dim.\n",
            "  'step':        [in] space between each scan point in each dim.\n",
            "  'group_size':  [in] min size of each group of points visisted first in a multi-dim loop.\n",
            "  'start':       [out] set to first scan point in called function(s) in inner loop(s).\n",
            "  'stop':        [out] set to one past last scan point in called function(s) in inner loop(s).\n",
            "  'index':       [out] set to zero on first iteration of loop; increments each iteration.\n",
            "  Each called function has a 'ScanIndices' variable as a parameter.\n",
            "  Values in the 'in' arrays in all dimensions are copied from the input.\n",
            "  Values in the 'out' arrays in any dimension not scanned are copied from the input.\n",
            "  Each array should be the length specified by the largest index used (typically same as -ndims).\n",
            "  The 'ScanIndices' input var is named with the -inVar option.\n",
            "Options:\n";
        print_options_help(\@KNOBS);
        print "Examples:\n",
            "  $script -ndims 2 'loop(0,1) { calc(f); }'\n",
            "  $script -ndims 3 'omp loop(0,1) { loop(2) { calc(f); } }'\n",
            "  $script -ndims 3 'omp loop(0,1) { prefetch loop(2) { calc(f); } }'\n",
            "  $script -ndims 3 'omp loop(0) { loop(1) { prefetch loop(2) { calc(f); } } }'\n",
            "  $script -ndims 3 'grouped omp loop(0..N-1) { calc(f); }'\n",
            "  $script -ndims 3 'omp loop(0) { serpentine loop(1..N-1) { calc(f); } }'\n",
            "  $script -ndims 4 'omp loop(0..N+1) { serpentine loop(N+2,N-1) { calc(f); } }'\n";
        exit 1;
    }

    @dims = 0 .. ($OPT{ndims} - 1);
    warn "info: generating scanning code for ".scalar(@dims)."-D grids...\n";
    $inputVar = $OPT{inVar};

    my $codeString = join(' ', @ARGV); # just concat all non-options params together.
    processCode($codeString);
}

main();
