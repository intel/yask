#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2021, Intel Corporation
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

use strict;
use File::Basename;
use File::Path;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

use File::Which;
use Text::ParseWords;
use FileHandle;
use CmdLine;

$| = 1;                         # autoflush.

# Globals.
my %OPT;                        # cmd-line options.
my @dims;                       # indices of dimensions.
my $inputVar;                   # input var.
my $outputVar;                  # output var.
my $loopPart = "USE_LOOP_PART_";
my @exprs = ("stride", "align", "align_ofs", "group_size");
my $indent = dirname($0)."/yask_indent.sh";

# loop-feature bit fields.
my $bSerp = 0x1;                # serpentine path
my $bSquare = 0x2;              # square_wave path
my $bGroup = 0x4;               # group path
my $bSimd = 0x8;                # simd prefix

##########
# Function to make names of variables based on dimension string(s).

# 'idx()' => "".
# 'idx(3)' => "[3]".
# 'idx(3,5)' => "[3][5]".
sub idx {
    return join('', map("[$_]", @_));
}

# Accessors for input struct.
# Examples if $inputVar == "block_idxs":
# inVar() => "block_idxs".
# inVar("foo") => "block_idxs.foo".
# inVar("foo", 5) => "block_idxs.foo[5]"
#                 OR "FOO_EXPR(5)" if @exprs contains "foo".
sub inVar {
    my $vname = shift;
    my $part = (defined $vname) ? ".$vname" : "";
    if (defined $vname && scalar @_ == 1 && grep( /^$vname$/, @exprs)) {
        my $em = (uc $vname)."_EXPR";
        return "$em(@_)";
    }
    return "$inputVar$part".idx(@_);
}

# Accessors for local (output) struct.
# Examples if $outputVar == "local_idxs":
# locVar() => "local_indices".
# locVar("foo", 5) => "local_indices.foo[5]".
sub locVar {
    my $vname = shift;
    my $part = (defined $vname) ? ".$vname" : "";
    return "$outputVar$part".idx(@_);
}

# Access values in input struct.
sub beginVar {
    return inVar("begin", @_);
}
sub endVar {
    return inVar("end", @_);
}
sub strideVar {
    return inVar("stride", @_);
}
sub alignVar {
    return inVar("align", @_);
}
sub alignOfsVar {
    return inVar("align_ofs", @_);
}
sub groupSizeVar {
    return inVar("group_size", @_);
}

# These are generated scalars.
sub adjAlignVar {
    return join('_', 'adj_align', @_);
}
sub alignBeginVar {
    return join('_', 'aligned_begin', @_);
}
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

# set var for the body.
sub setOutVar {
    my @dims = @_;

    my @stmts;
    map {
        push @stmts,
            " ".locVar("start", $_)." = ".startVar($_).";",
            " ".locVar("stop", $_)." = ".stopVar($_).";",
            " ".locVar("cur_indices", $_)." = ".indexVar($_).";";
    } @dims;
    return @stmts;
}

###########
# Loop-constructing functions.

# return type of var needed for loop index.
# args: dimension(s) -- currently ignored.
sub indexType {
    return 'idx_t';
}

# Create and init vars *before* beginning of simple or collapsed loop.
sub addIndexVars1($$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions.
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
                my $bvar = beginVar($dim);
                my $evar = endVar($dim);
                my $svar = strideVar($dim);
                my $avar = alignVar($dim);
                my $aovar = alignOfsVar($dim);
                my $aavar = adjAlignVar($dim);
                my $abvar = alignBeginVar($dim);
                my $nvar = numItersVar($dim);
                my $ntvar = numGroupsVar($dim);
                my $tsvar = groupSizeVar($dim);
                my $ntivar = numFullGroupItersVar($dim);

                # Example alignment:
                # bvar = 20.
                # svar = 8.
                # avar = 4.
                # aovar = 15.
                # Then,
                # aavar = min(4, 8) = 4.
                # abvar = round_down_flr(20 - 15, 4) + 15 = 4 + 15 = 19.

                push @$code,
                    " // Alignment must be less than or equal to stride size.",
                    " const $itype $aavar = std::min($avar, $svar);",
                    " // Aligned beginning point such that ($bvar - $svar) < $abvar <= $bvar.",
                    " const $itype $abvar = yask::round_down_flr($bvar - $aovar, $aavar) + $aovar;",
                    " // Number of iterations to get from $abvar to (but not including) $evar, striding by $svar.".
                    " This value is rounded up because the last iteration may cover fewer than $svar strides.",
                    " const $itype $nvar = yask::ceil_idiv_flr($evar - $abvar, $svar);";

                # For grouped loops.
                if ($features & $bGroup) {

                    # loop iterations within one group.
                    push @$code,
                        " // Number of iterations in one full group in dimension $dim.".
                        " This value is rounded up, effectively increasing the group size if needed".
                        " to a multiple of $svar.".
                        " A group is considered 'full' if it has the max number of iterations.",
                        " const $itype $ntivar = std::min(yask::ceil_idiv_flr($tsvar, $svar), $nvar);";

                    # number of full groups.
                    push @$code, 
                        " // Number of full groups in dimension $dim.",
                        " const $itype $ntvar = $ntivar ? $nvar / $ntivar : 0;";
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
sub addIndexVars2($$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions in loop.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack, including enclosing dims.

    my $itype = indexType(@$loopDims);
    my $civar = loopIndexVar(@$loopDims); # collapsed index var; everything based on this.
    my $outerDim = $loopDims->[0];        # outer dim of these loops.
    my $innerDim = $loopDims->[$#$loopDims]; # inner dim of these loops.

    # Grouping.
    if ($features & $bGroup) {

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
            my $tival = "$tgvar ? $prevOvar / $tgvar : 0";
            push @$code,
                " // Index of this group in dimension $dim.",
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
                " // Adjust number of iterations in this group in dimension $dim.",
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
                " // Offset within this group in dimension $dim.",
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
        my $innerDivar = indexVar($innerDim);
        my $innerNvar = numItersVar($innerDim);

        # loop through each dim, outer to inner.
        for my $i (0..$#$loopDims) {
            my $dim = $loopDims->[$i];
            my $nvar = numItersVar($dim);
            my $isInner = ($i == $#$loopDims);

            # Goal is to compute $divar from 1D $ivar.
            my $ivar = $civar;
            my $divar = indexVar($dim);

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
                " // Zero-based, unit-stride index for ".dimStr($dim).".",
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

    # start and stop vars based on individual begin, end, stride, and index vars.
    for my $dim (@$loopDims) {
        my $divar = indexVar($dim);
        my $stvar = startVar($dim);
        my $spvar = stopVar($dim);
        my $bvar = beginVar($dim);
        my $abvar = alignBeginVar($dim);
        my $evar = endVar($dim);
        my $svar = strideVar($dim);
        push @$code,
            " // This value of $divar covers ".dimStr($dim)." from $stvar to (but not including) $spvar.",
            " idx_t $stvar = std::max($abvar + ($divar * $svar), $bvar);",
            " idx_t $spvar = std::min($abvar + (($divar+1) * $svar), $evar);";
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
    addIndexVars2($code, $loopDims, $features, $loopStack);
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
        push @toks, $tok unless $tok =~ /^\s+$/;
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
          return 0;             # starting another loop, so not inner.
      }
      elsif ($tok eq '}') {
          return 1;             # end of loop, so is inner.
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
        elsif ($arg =~ /^\.\.+$/) {
            die "Error: missing token before '$arg'.\n"
                if !defined $prevArg;
            die "Error: non-numerical token before '$arg'.\n"
                if $prevArg !~ /^[-]?\d+$/;
            pop @args;
            my $arg2 = getNextArg($toks, $ti);
            die "Error: missing token after '$arg'.\n"
                if !defined $arg2;
            die "Error: non-numerical token after '$arg'.\n"
                if $arg2 !~ /^[-]?\d+$/;
            for my $i ($prevArg .. $arg2) {
                push @args, $i;
            }
        }

        else {
            die "Error: non-numerical token '$arg'.\n"
                if $arg !~ /^[-]?\d+$/;
            push @args, $arg;
            $prevArg = $arg;
        }
    }
    return @args;
}

# Generate a pragma w/given text.
sub pragma($) {
    my $codeString = shift;
    return (length($codeString) > 0) ?
        "_Pragma(\"$codeString\")" : "";
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
    my @loopDims;               # dimension(s) of current loop.
    my $curInnerDim;            # iteration dimension of inner loop (undef if not in inner loop).
    my $innerNum = 0;           # inner-loop counter.

    # modifiers before loop() statements.
    my @loopPrefix;             # string(s) to put before loop body.
    my $features = 0;           # bits for loop features.

    # Lines of code to output.
    my @code;

    # Front matter.
    push @code,
        "#ifdef $loopPart$innerNum\n",
        "// These macros must be re-defined for each generated loop-nest.",
        "#ifndef OMP_PRAGMA",
        "#define OMP_PRAGMA ".pragma($OPT{ompMod}),
        "#endif",
        "#ifndef SIMD_PRAGMA",
        "#define SIMD_PRAGMA ".pragma($OPT{simdMod}),
        "#endif",
        "#ifndef INNER_PRAGMA",
        "#define INNER_PRAGMA ".pragma($OPT{innerMod}),
        "#endif";
    for my $expr (@exprs) {
        my $em = (uc $expr)."_EXPR";
        push @code,
            "#ifndef $em",
            "#define ${em}(dim_num) ".inVar($expr).'[dim_num]',
            "#endif";
    }
    push @code,
        "// 'ScanIndices $inputVar' must be set before the following code.",
        "{";
    
    # loop thru all the tokens ni the input.
    for (my $ti = 0; $ti <= $#toks; ) {
        my $tok = checkToken($toks[$ti++], '.*', 1);

        # use OpenMP on next loop.
        if (lc $tok eq 'omp') {

            push @loopPrefix,
                " // Distribute iterations among OpenMP threads.", 
                " OMP_PRAGMA";
            print "info: using OpenMP on following loop.\n";
        }

        # generate simd in next loop.
        elsif (lc $tok eq 'simd') {

            push @loopPrefix, ' OMP_SIMD';
            $features |= $bSimd;
            print "info: generating SIMD in following loop.\n";
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
            @loopDims = getArgs(\@toks, \$ti);  # might be empty.
            checkToken($toks[$ti++], '\{', 1); # eat the '{'.
            my $ndims = scalar(@loopDims);

            # check for existence of all vars.
            for my $ld (@loopDims) {
                die "Error: loop variable '$ld' not in ".dimStr(@dims).".\n"
                    if !grep($_ eq $ld, @dims);
                for my $ls (@loopStack) {
                    die "Error: loop variable '$ld' already used.\n"
                        if $ld == $ls;
                }
            }

            push @loopStack, @loopDims;          # all dims so far.
            push @loopCounts, $ndims; # number of dims in each loop.

            # set inner dim if applicable.
            undef $curInnerDim;
            if ($ndims && isInInner(\@toks, \$ti)) {
                $curInnerDim = $loopDims[$#loopDims];
            }

            # Start the loop unless it's an inner loop.
            if ($ndims) {
                print "info: generating scan over ".dimStr(@loopDims)."...\n";

                # add initial code for index vars, but don't start loop body yet.
                addIndexVars1(\@code, \@loopDims, $features);
            
                # if *not* the inner loop, start the loop body.
                # if it is the inner loop, we might need more than one loop body, so
                # it will not be generated until the '}' is seen.
                if (!defined $curInnerDim) {
                    beginLoop(\@code, \@loopDims, \@loopPrefix, 0, undef, $features, \@loopStack);

                    # clear data for this loop.
                    undef @loopDims;
                    undef @loopPrefix;
                    $features = 0;
                }
            } else {
                print "info: generating dummy loop for empty $tok args...\n";
                my $ivar = "dummy" . scalar(@loopCounts);
                push @code, " // Dummy loop.",
                    @loopPrefix,
                    " for (int $ivar = 0; $ivar < 1; $ivar++) {";

                # clear data for this loop.
                undef @loopDims;
                undef @loopPrefix;
                $features = 0;
            }

        }

        # End of loop.
        # This is where most of @code is created for inner loops.
        elsif ($tok eq '}') {
            die "error: attempt to end loop w/o beginning\n" if !@loopCounts;

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
                my $beginVal = 0;
                my $endVal = numItersVar(@loopDims);
                my $comment = " // Inner loop.";

                # beginning of loop.
                push @code, $comment;
                push @code, " INNER_PRAGMA";
                beginLoop(\@code, \@loopDims, \@loopPrefix, 
                          $beginVal, $endVal, $features, \@loopStack);

                # Indices for body.
                push @code, 
                    " // Local copy of indices for loop body.",
                    " ScanIndices ".locVar()."($inputVar);",
                    setOutVar(@loopStack);

                # Break for body.
                push @code,
                    "#undef $loopPart$innerNum\n",
                    "#endif\n",
                    "\n// Loop body goes here.\n\n";

                $innerNum++;
                push @code,
                    "#ifdef $loopPart$innerNum\n";

                # end of loop.
                endLoop(\@code);

                # clear data for this loop.
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

    # Back matter.
    push @code,
        "}",
        "#undef OMP_PRAGMA",
        "#undef INNER_PRAGMA",
        "#undef SIMD_PRAGMA";
    for my $expr (@exprs) {
        my $em = (uc $expr)."_EXPR";
        push @code,
            "#undef $em";
    }
    push @code,
        "#undef $loopPart$innerNum\n",
        "#endif\n";
    
    # open output stream.
    open OUT, "> $OPT{output}" or die;

    # header.
    my $ndims = scalar(@dims);
    print OUT
        "/*\n",
        " * $ndims-D var-scanning code.\n",
        " * Generated automatically from the following pseudo-code:\n",
        " *\n",
        " * N = ",$ndims,";\n",
        " * $codeString\n",
        " *\n */";

    # print out code.
    for my $line (@code) {
        print OUT "\n" if $line =~ m=^\s*//=; # blank line before comment.
        print OUT " $line\n";
    }
    print OUT "// End of generated code.";
    close OUT;
    system("$indent $OPT{output}") if -x $indent;

    print "info: output in '$OPT{output}'.\n";
}

# Parse arguments and emit code.
sub main() {

    my(@KNOBS) = (
        # knob,        description,   optional default
        [ "ndims=i", "Value of N.", 1],
        [ "inVar=s", "Name of existing input 'ScanIndices' var.", 'loop_indices'],
        [ "outVar=s", "Name of created loop-body 'ScanIndices' var.", 'body_indices'],
        [ "ompMod=s", "Set OMP_PRAGMA to insert before 'omp' loop(s).", "omp parallel for"],
        [ "simdMod=s", "Set SIMD_PRAGMA to insert before 'simd' loop(s).", "omp simd"],
        [ "innerMod=s", "Set INNER_PRAGMA to insert before inner loop(s).", ''],
        [ "output=s", "Name of output file.", 'loops.h'],
        );
    my($command_line) = process_command_line(\%OPT, \@KNOBS);
    print "$command_line\n" if $OPT{verbose};

    my $script = basename($0);
    if (!$command_line || $OPT{help} || @ARGV < 1) {
        print "Outputs C++ code to scan N-D vars.\n",
            "Usage: $script [options] <code-string>\n",
            "The <code-string> contains optionally-nested scans across the given\n",
            "  indices between 0 and N-1 indicated by 'loop(<indices>)'\n",
            "Indices may be specified as a comma-separated list or <first..last> range,\n",
            "  using the variable 'N' as needed.\n",
            "A loop statement with more than index will generate a single collapsed loop.\n",
            "The generated code will contain a macro-delimited part before the first inner loop body\n",
            "  and a part after each inner loop body.\n",
            "Optional loop modifiers:\n",
            "  omp:             add OMP_PRAGMA to loop (distribute work across SW threads).*\n",
            "  simd:            add SIMD_PRAMA to loop (distribute work across SIMD HW).*\n",
            "  grouped:         generate grouped scan within a collapsed loop.\n",
            "  serpentine:      generate reverse scan when enclosing loop dimension is odd.*\n",
            "  square_wave:     generate 2D square-wave scan for two innermost dimensions of a collapsed loop.*\n",
            "      * Do not use these modifiers for YASK rank or block loops because they must\n",
            "        execute with strictly-increasing indices when using temporal tiling.\n",
            "A 'ScanIndices' type must be defined in C++ code prior to including the generated code.\n",
            "  This struct contains the following 'Indices' elements:\n",
            "  'begin':       [in] first index to scan in each dim.\n",
            "  'end':         [in] value beyond last index to scan in each dim.\n",
            "  'stride':      [in] distance between each scan point in each dim.\n",
            "  'align':       [in] alignment of strides after first one.\n",
            "  'align_ofs':   [in] value to subtract from 'start' before applying alignment.\n",
            "  'group_size':  [in] min size of each group of points visisted first in a multi-dim loop.\n",
            "  'start':       [out] set to first scan point in body of inner loop(s).\n",
            "  'stop':        [out] set to one past last scan point in body of inner loop(s).\n",
            "  'index':       [out] set to zero on first iteration of loop; increments each iteration.\n",
            "  Each array should be the length specified by the largest index used (typically same as -ndims).\n",
            "A 'ScanIndices' input var must be defined in C++ code prior to including the generated code.\n",
            "  The 'in' indices control the range of the generaged loop(s).\n",
            "  The 'ScanIndices' input var is named with the -inVar option.\n",
            "Loop indices will be available in the body of the loop in a new 'ScanIndices' var.\n",
            "  Values in the 'in' indices are copied from the input var.\n",
            "  Values in the 'out' indices are set to indicate the range to be covered by the loop body.\n",
            "  Values in the 'out' indices in any dim not scanned are copied from the input.\n",
            "  The 'ScanIndices' output var is named with the -outVar option.\n",
            "Options:\n";
        print_options_help(\@KNOBS);
        print "Examples:\n",
            "  $script -ndims 2 'loop(0,1) { }'\n",
            "  $script -ndims 3 'omp loop(0,1) { loop(2) { } }'\n",
            "  $script -ndims 3 'omp loop(0) { loop(1,2) { } }'\n",
            "  $script -ndims 3 'grouped omp loop(0..N-1) { }'\n",
            "  $script -ndims 3 'omp loop(0) { square_wave loop(1..N-1) { } }'\n",
            "  $script -ndims 4 'omp loop(0..N-2) { loop(N-2,N-1) { } }'\n";
        exit 1;
    }

    @dims = 0 .. ($OPT{ndims} - 1);
    print "info: generating scanning code for ".scalar(@dims)."-D vars...\n";
    $inputVar = $OPT{inVar};
    $outputVar = $OPT{outVar};

    my $codeString = join(' ', @ARGV); # just concat all non-options params together.
    processCode($codeString);
}

main();
