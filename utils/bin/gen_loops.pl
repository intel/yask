#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2024, Intel Corporation
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

# Purpose: Create area-scanning code.

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

##########
# Globals.
my %OPT;                        # cmd-line options.
my %macros;                     # macros from file.
my $inputVar = "LOOP_INDICES";  # input var macro.
my $outputVar = "BODY_INDICES"; # output var.
my $loopPart = "USE_LOOP_PART_"; # macro to enable specified loop part.
my $macroPrefix = "";           # prefix for macros.
my $varPrefix = "";             # prefix for vars.
my $doAlign = 1;                # generate alignment code.
my @fixed_exprs = ("begin", "end", "stride", "tile_size");
my @align_exprs = ("align", "align_ofs");
my @var_exprs = ("start", "stop", "index");
my $indent = dirname($0)."/yask_indent.sh";

# loop-feature bit fields.
my $bSerp = 0x1;                # serpentine path
my $bSquare = 0x2;              # square_wave path
my $bTile = 0x4;                # tile path
my $bOmpPar = 0x8;              # OpenMP parallel
my $bManual = 0x10;             # use manual scheduling
my $bNested = 0x20;             # use "normal" nested loops
my $bSimd = 0x40;               # OpenMP SIMD

##########
# Various functions to create variable references.

# Create indices from args.
# 'idx()' => "".
# 'idx(3)' => "[3]".
# 'idx(3,5)' => "[3][5]".
sub idx {
    return join('', map("[$_]", @_));
}

# Accessors for input struct.
# Examples if $inputVar == "block_idxs":
# inVar() => "block_idxs".
# inVar("foo", 5) => "FOO(5)" (using macro).
sub inVar {
    my $vname = shift;
    if (defined $vname) {
        die unless scalar(@_) == 1;
        my $em = $macroPrefix.(uc $vname);
        return "$em(@_)";
    }
    return "$macroPrefix$inputVar";
}

# Accessors for output struct.
# Examples if $outputVar == "local_indices":
# outVar() => "local_indices".
# outVar("foo", 5) => "local_indices.foo[5]".
sub outVar {
    my $vname = shift;
    if (defined $vname) {
        die unless scalar(@_) == 1;
        return "$macroPrefix$outputVar.$vname".idx(@_);
    }
    return "$macroPrefix$outputVar";
}

# Make a local var.
sub locVar {
    return $varPrefix . join('_', @_);
}

# Names for vars used in the generated code.
# Arg(s) are loop dim(s).
sub beginVar {
    return locVar("begin", @_);
}
sub endVar {
    return locVar("end", @_);
}
sub strideVar {
    return locVar("stride", @_);
}
sub alignVar {
    return locVar("align", @_);
}
sub alignOfsVar {
    return locVar("align_ofs", @_);
}
sub tileSizeVar {
    return locVar("tile_size", @_);
}
sub adjAlignVar {
    return locVar('adj_align', @_);
}
sub alignBeginVar {
    return locVar('aligned_begin', @_);
}
sub numItersVar {
    return locVar('num_iters', @_);
}
sub numTilesVar {
    return locVar('num_full_tiles', @_);
}
sub numFullTileItersVar {
    return locVar('num_iters_in_full_tile', @_);
}
sub numTileSetItersVar {
    return scalar @_ ? locVar('num_iters_in_tile_set', @_) :
        locVar('num_iters_in_full_tile');
}
sub indexVar {
    return locVar('index', @_);
}
sub tileIndexVar {
    return locVar('index_of_tile', @_);
}
sub tileSetOffsetVar {
    return scalar @_ ? locVar('index_offset_within_tile_set', @_) :
        locVar('index_offset_within_this_tile');
}
sub tileOffsetVar {
    return locVar('index_offset_within_this_tile', @_);
}
sub numLocalTileItersVar {
    return locVar('num_iters_in_tile', @_);
}
sub loopIndexVar {
    return locVar('loop_index', @_);
}
sub startVar {
    return locVar('start', @_);
}
sub stopVar {
    return locVar('stop', @_);
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

# Conditionally define a macro.
sub macroDef($$$) {
    my $mname = shift;
    my $margs = shift;
    my $mdef = shift;

    $mname = uc $mname;
    $margs = (defined $margs) ? "($margs)" : "";
    return
        "#ifndef ${macroPrefix}$mname",
        "#define ${macroPrefix}$mname$margs $mdef",
        "#endif";
}
sub macroUndef($) {
    my $mname = shift;

    $mname = uc $mname;
    return
        "#ifdef ${macroPrefix}$mname",
        "#undef ${macroPrefix}$mname",
        "#endif";
}

# copy vars from the input.
sub getInVars {
    my $tiledDims = shift;      # ref to hash.
    my @ldims = @_;

    my $itype = indexType();
    my @stmts;
    for my $dim (@ldims) {

        # Vars for input values.
        my $bvar = beginVar($dim);
        my $evar = endVar($dim);
        my $svar = strideVar($dim);
        my $tsvar = tileSizeVar($dim);
        my $avar = alignVar($dim);
        my $aovar = alignOfsVar($dim);
        push @stmts,
            "// Create input vars for dim $dim.",
            "const $itype $bvar = ".inVar("begin", $dim).";",
            "const $itype $evar = ".inVar("end", $dim).";",
            "const $itype $svar = ".inVar("stride", $dim).";";
        push @stmts,
            "const $itype $avar = ".inVar("align", $dim).";",
            "const $itype $aovar = ".inVar("align_ofs", $dim).";"
            if $doAlign;
        push @stmts,
            "const $itype $tsvar = ".inVar("tile_size", $dim).";"
            if defined $$tiledDims{$dim};
    }
    return @stmts;
}

# make macros for the body.
sub makeOutMacros {
    my @ldims = @_;

    my @stmts;
    for my $expr (@var_exprs) {
        my $base = "BODY_".uc($expr);
        push @stmts, macroDef($base, "dim_num", "YCAT(".locVar($expr)."_, dim_num)");
    }

    return @stmts;
}

# set var for the body.
sub setOutVars {
    my @ldims = @_;

    my $itype = indexType();
    my @stmts;

    for my $expr (@var_exprs) {
        for my $dim (@ldims) {
            my $macro = "${macroPrefix}BODY_".uc($expr)."($dim)";
            push @stmts, outVar($expr, $dim)." = $macro;";
        }
    }
    return @stmts;
}

###########
# Loop-constructing functions.

# return type of var needed for loop index.
sub indexType() {
    return 'idx_t';
}

# Adjust features.
# Returns new set of features.
sub adjFeatures($$$) {
    my $loopDims = shift;       # ref to list of dimensions in loop.
    my $features = shift;             # feature bits for path types.
    my $loopStack = shift;      # whole stack at this point, including enclosing dims.

    my $ndims = scalar @$loopDims;
    my $outerDim = $loopDims->[0];        # outer dim of these loops.
    my $innerDim = $loopDims->[$#$loopDims]; # inner dim of these loops.

    # find enclosing dim outside of these loops if avail.
    my $encDim;
    map { $encDim = $loopStack->[$_]
              if $loopStack->[$_ + 1] eq $outerDim; } 0..($#$loopStack-1);

    if (($features & $bManual) && !($features & $bOmpPar)) {
        warn "notice: manual ignored for non-OpenMP parallel loop.\n";
        $features &= ~$bManual;    # clear bits.
    }
    if ($ndims < 2 && ($features & ($bSquare | $bNested))) {
        warn "notice: square-wave and nested ignored for loop with only $ndims dim.\n";
        $features &= ~($bSquare | $bNested);     # clear bits.
    }
    if ($ndims < 2 && !defined $encDim && ($features & $bSerp)) {
        warn "notice: serpentine ignored for outer loop.\n";
        $features &= ~$bSerp;     # clear bit.
    }
    
    if ($features & $bTile) {

        if ($ndims < 2) {
            warn "notice: tiling ignored for loop with only $ndims dim.\n";
            $features &= ~$bTile;     # clear bit.
        }
        die "error: serpentine not compatible with tiling.\n"
            if $features & $bSerp;
        die "error: square-wave not compatible with tiling.\n"
            if $features & $bSquare;
    }
    if ($features & ($bManual | $bNested)) {
        die "error: serpentine not compatible with manual or nested.\n"
            if $features & $bSerp;
        die "error: square-wave not compatible with manual or nested.\n"
            if $features & $bSquare;
        die "error: tiling not compatible with manual or nested.\n"
            if $features & $bTile;
    }
    return $features;
}

# Create and init vars *before* beginning of loop(s) in given dim(s).
# These compute loop-invariant values like number of iterations.
sub addIndexVars1($$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions in this loop.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack at this point, including enclosing dims.

    push @$code,
        "// ** Begin scan over ".dimStr(@$loopDims).". **";

    my $itype = indexType();

    for my $pass (0..1) {
        for my $i (0..$#$loopDims) {
            my $dim = $loopDims->[$i];
            my $isInner = ($i == $#$loopDims);

            # Pass 0: iterations.
            if ($pass == 0) {

                # Vars from the struct.
                my $bvar = beginVar($dim);
                my $evar = endVar($dim);
                my $svar = strideVar($dim);
                my $avar = alignVar($dim);
                my $aovar = alignOfsVar($dim);
                my $tsvar = tileSizeVar($dim);

                # New vars.
                my $aavar = adjAlignVar($dim);
                my $abvar = alignBeginVar($dim);
                my $nvar = numItersVar($dim);
                my $ntvar = numTilesVar($dim);
                my $ntivar = numFullTileItersVar($dim);

                # Example alignment:
                # bvar = 20.
                # svar = 8.
                # avar = 4.
                # aovar = 15.
                # Then,
                # aavar = min(4, 8) = 4.
                # abvar = round_down_flr(20 - 15, 4) + 15 = 4 + 15 = 19.

                if ($doAlign) {
                    push @$code,
                        "// Alignment must be less than or equal to stride size.",
                        "const $itype $aavar = std::min($avar, $svar);",
                        "// Aligned beginning point such that ($bvar - $svar) < $abvar <= $bvar.",
                        "const $itype $abvar = yask::round_down_flr($bvar - $aovar, $aavar) + $aovar;",
                        "// Number of iterations to get from $abvar to (but not including) $evar, striding by $svar. ".
                        "This value is rounded up because the last iteration may cover fewer than $svar strides.",
                        "const $itype $nvar = yask::ceil_idiv_flr($evar - $abvar, $svar);";
                } else {
                    push @$code,
                        "// Number of iterations to get from $bvar to (but not including) $evar, striding by $svar. ".
                        "This value is rounded up because the last iteration may cover fewer than $svar strides.",
                        "const $itype $nvar = yask::ceil_idiv_flr($evar - $bvar, $svar);";
                }

                # For tiled loops.
                if ($features & $bTile) {

                    # loop iterations within one tile.
                    push @$code,
                        "// Number of iterations in one full tile in dimension $dim.".
                        "This value is rounded up, effectively increasing the tile size if needed".
                        "to a multiple of $svar.".
                        "A tile is considered 'full' if it has the max number of iterations.",
                        "const $itype $ntivar = std::min(yask::ceil_idiv_flr($tsvar, $svar), $nvar);";

                    # number of full tiles.
                    push @$code, 
                        "// Number of full tiles in dimension $dim.",
                        "const $itype $ntvar = $ntivar ? $nvar / $ntivar : 0;";
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
                    "// Number of iterations in $loopStr",
                    "const $itype $snvar = $snval;";
            }
        }
    }
}

# Add index variables *inside* the loop.
sub addIndexVars2($$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions in loop.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack at this point, including enclosing dims.

    my $itype = indexType();
    my $civar = loopIndexVar(@$loopDims); # multi-dim index var; everything based on this.
    my $ndims = scalar @$loopDims;
    my $outerDim = $loopDims->[0];        # outer dim of these loops.
    my $innerDim = $loopDims->[$#$loopDims]; # inner dim of these loops.

    # Tiling.
    if ($features & $bTile) {

        # declare local size vars.
        push @$code,
            "// Working vars for iterations in tiles.".
            "These are initialized to full-tile counts and then".
            "reduced if/when in a partial tile.";
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $ltvar = numLocalTileItersVar($dim);
            my $ltval = numFullTileItersVar($dim);
            push @$code, "$itype $ltvar = $ltval;";
        }

        # calculate tile indices and sizes and 1D offsets within tiles.
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

            # Size of tile set.
            my $tgvar = numTileSetItersVar(@inDims);
            my $tgval = join(' * ', 
                             (map { numLocalTileItersVar($_) } @dims),
                             (map { numItersVar($_) } @inDims));
            my $tgStr = @inDims ?
                "the set of tiles across $inStr" : "this tile";
            push @$code,
                "// Number of iterations in $tgStr.",
                "$itype $tgvar = $tgval;";

            # Index of this tile in this dim.
            my $tivar = tileIndexVar($dim);
            my $tival = "$tgvar ? $prevOvar / $tgvar : 0";
            push @$code,
                "// Index of this tile in dimension $dim.",
                "$itype $tivar = $tival;";

            # 1D offset within tile set.
            my $ovar = tileSetOffsetVar(@inDims);
            my $oval = "$prevOvar % $tgvar";
            push @$code,
                "// Linear offset within $tgStr.",
                "$itype $ovar = $oval;";

            # Size of this tile in this dim.
            my $ltvar = numLocalTileItersVar($dim);
            my $ltval = numItersVar($dim).
                " - (".numTilesVar($dim)." * ".numFullTileItersVar($dim).")";
            push @$code,
                "// Adjust number of iterations in this tile in dimension $dim.",
                "if ($tivar >= ".numTilesVar($dim).")".
                "  $ltvar = $ltval;";

            # for next dim.
            $prevOvar = $ovar;
        }

        # Calculate nD indices within tile and overall.
        # TODO: allow different paths *within* tile.
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $tivar = tileIndexVar($dim);
            my $ovar = tileSetOffsetVar(); # last one calculated above.

            # dims after (inside of) $i (empty for inner dim)
            my @inDims = @$loopDims[$i + 1 .. $ndims - 1];

            # Determine offset within this tile.
            my $dovar = tileOffsetVar($dim);
            my $doval = $ovar;

            # divisor of index is product of sizes of remaining nested dimensions.
            if (@inDims) {
                my $subVal = join(' * ', map { numLocalTileItersVar($_) } @inDims);
                $doval .= " / ($subVal)";
            }

            # mod by size of this dimension (not needed for outer-most dim).
            if ($i > 0) {
                $doval = "($doval) % ".numLocalTileItersVar($dim);
            }

            # output offset in this dim.
            push @$code,
                "// Offset within this tile in dimension $dim.",
                "$itype $dovar = $doval;";

            # final index in this dim.
            my $divar = indexVar($dim);
            my $dival = numFullTileItersVar($dim)." * $tivar + $dovar";
            push @$code,
                "// Zero-based, unit-stride index for ".dimStr($dim).".",
                "$itype $divar = $dival;";
        }
    }

    # No tiling.
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
                "// Zero-based, unit-stride index for ".dimStr($dim).".",
                "$itype $divar = $dival;";

            # apply square-wave to inner 2 dimensions if requested.
            my $isInnerSquare = $ndims >=2 && $isInner && ($features & $bSquare);
            if ($isInnerSquare) {

                my $divar2 = "index_x2";
                my $bvar = "lsb";
                push @$code, 
                    "// Modify $prevDivar and $divar for 'square_wave' path.",
                    "if (($innerNvar > 1) && ($prevDivar/2 < $prevNvar/2)) {",
                    "  // Compute extended index over 2 iterations of $prevDivar.",
                    "  $itype $divar2 = $divar + ($nvar * ($prevDivar & 1));",
                    "  // Select $divar from 0,0,1,1,2,2,... sequence",
                    "  $divar = $divar2 / 2;",
                    "  // Select $prevDivar adjustment value from 0,1,1,0,0,1,1, ... sequence.",
                    "  $itype $bvar = ($divar2 & 0x1) ^ (($divar2 & 0x2) >> 1);",
                    "  // Adjust $prevDivar +/-1 by replacing bit 0.",
                    "  $prevDivar = ($prevDivar & $itype(-2)) | $bvar;",
                    "} // square-wave.";
            }

            # reverse order of every-other traversal if requested.
            # for inner dim with square-wave, do every 2.
            if (($features & $bSerp) && defined $prevDivar) {
                if ($isInnerSquare) {
                    push @$code,
                        "// Reverse direction of $divar after every-other iteration of $prevDivar for 'square_wave serpentine' path.",
                        "if (($prevDivar & 2) == 2) $divar = $nvar - $divar - 1;";
                } else {
                    push @$code,
                        "// Reverse direction of $divar after every iteration of $prevDivar for  'serpentine' path.",
                        "if (($prevDivar & 1) == 1) $divar = $nvar - $divar - 1;";
                }
            }

            $prevDim = $dim;
            $prevDivar = $divar;
            $prevNvar = $nvar;
        }
    }
}

# Add start/stop variables *inside* the loop.
sub addIndexVars3($$$$) {
    my $code = shift;           # ref to list of code lines.
    my $loopDims = shift;       # ref to list of dimensions in loop.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack at this point, including enclosing dims.

    my $itype = indexType();

    # start and stop vars based on individual begin, end, stride, and index vars.
    for my $dim (@$loopDims) {
        my $divar = indexVar($dim);
        my $stvar = startVar($dim);
        my $spvar = stopVar($dim);
        my $bvar = beginVar($dim);
        my $abvar = alignBeginVar($dim);
        my $evar = endVar($dim);
        my $svar = strideVar($dim);
        if ($doAlign) {
            push @$code,
                "// This value of $divar covers ".dimStr($dim)." from $stvar to (but not including) $spvar.",
                "$itype $stvar = std::max($abvar + ($divar * $svar), $bvar);",
                "$itype $spvar = std::min($abvar + (($divar+1) * $svar), $evar);";
        } else {
            push @$code,
                "// This value of $divar covers ".dimStr($dim)." from $stvar to (but not including) $spvar.",
                "$itype $stvar = $bvar + ($divar * $svar);",
                "$itype $spvar = std::min($bvar + (($divar+1) * $svar), $evar);";
        }
    }
}

# Start of loop(s) over given dim(s).
# Every loop starts with 2 "{"'s to make ending easy.
# TODO: keep track of number of "{"'s used.
sub beginLoop($$$$$$$) {
    my $code = shift;           # ref to list of code lines to be added to.
    my $loopDims = shift;       # ref to list of dimensions for this loop.
    my $features = shift;       # bits for path types.
    my $loopStack = shift;      # whole stack of dims so far, including enclosing dims.
    my $prefix = shift;         # ref to list of prefix code. May be undef.
    my $beginVal = shift;       # beginning of loop (undef for default).
    my $endVal = shift;         # end of loop (undef for default).

    $beginVal = 0 if !defined $beginVal;
    $endVal = numItersVar(@$loopDims) if !defined $endVal;
    $features = adjFeatures($loopDims, $features, $loopStack);
    my $itype = indexType();
    my $ivar = loopIndexVar(@$loopDims);
    my $ndims = scalar @$loopDims;

    # Add pre-loop index vars.
    addIndexVars1($code, $loopDims, $features, $loopStack);

    # Start "normal" nested loops.
    if ($features & $bNested) {
        push @$code, @$prefix if defined $prefix;
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $nvar = numItersVar($dim);
            my $divar = indexVar($dim);
            push @$code,
                "for ($itype $divar = 0; $divar < $nvar; $divar++)";
        }
        push @$code, "{ {";
    }

    # Start a parallel region if using manual distribution.
    elsif ($features & $bManual) {
        push @$code,
            "// Start parallel section.", @$prefix
            if defined $prefix;
        push @$code,
            "{",
            "// Number of threads in this parallel section.",
            "$itype nthreads = ${macroPrefix}OMP_NUM_THREADS;",
            "// Unique 0-based thread index in this parallel section.",
            "$itype thread_num = ${macroPrefix}OMP_THREAD_NUM;",
            "host_assert(thread_num < nthreads);",
            "// Begin and end indices for this thread.",
            "$itype thread_begin = $beginVal + ".
            "yask::div_equally_cumu_size_n($endVal - $beginVal, nthreads, thread_num - 1);",
            "$itype thread_end = $beginVal + ".
            "yask::div_equally_cumu_size_n($endVal - $beginVal, nthreads, thread_num);",
            "// Starting index.",
            "$itype $ivar = thread_begin;";

        # Add initial-loop index vars for this thread.
        addIndexVars2($code, $loopDims, $features, $loopStack);

        # Add sequential loops for this thread thread.
        # TODO: use one loop and increment-and-wrap-around code.
        push @$code,
            "\n // Loop through the ranges of $ndims dim(s) in this thread.";
        for my $i (0 .. $ndims-1) {
            my $dim = $loopDims->[$i];
            my $nvar = numItersVar($dim);
            my $divar = indexVar($dim);
            push @$code,
                "for (; $divar < $nvar && $ivar < thread_end; $divar++, ".
                ($i < $ndims-1 ? indexVar($loopDims->[$i+1])."=0" : "$ivar++").
                ")";
        }
        push @$code, " {";
    }
    
    # Start manually-flattened loop.
    else {
        push @$code, @$prefix if defined $prefix;
        push @$code,
            "for ($itype $ivar = $beginVal; $ivar < $endVal; $ivar++) { {";
        
        # Add inner-loop index vars for this iteration.
        addIndexVars2($code, $loopDims, $features, $loopStack);
    }

    # Add start/stop vars for this iteration.
    addIndexVars3($code, $loopDims, $features, $loopStack);
}

# End loops.
# Every loop ends with 2 "{"'s.
sub endLoop($) {
    my $code = shift;           # ref to list of code lines.

    push @$code, "} }";
}

##########
# Parsing functions.

# Split a string into tokens, ignoring whitespace.
sub tokenize($) {
    my $str = shift;

    # Find tokens.
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

  while (1) {
    my $tok = checkToken($toks->[$$ti++], '\w+|\,|\.+|\)', 1);

    # comma (ignore).
    if ($tok eq ',') {
    }

    # end (done).
    elsif ($tok eq ')') {
      return undef;
    }

    # actual token.
    else {
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
            if ($prevArg == $arg2) {
                push @args, $arg2;
            }
            elsif ($prevArg < $arg2) {
                for my $i ($prevArg .. $arg2) {
                    push @args, $i;
                }
            }
            else {
                # Something like 2..1, so return empty list.
                # TODO: add an operator to allow reverse ordering.
            }
        }

        # Should be a number.
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

    # vars across loops.
    my $partNum = 0;            # macro-part counter.
    my %dims;                   # all dims seen.
    my %tiledDims;              # dims needing tiles.
    my @code;                   # code to output.

    # vars to track one loop.
    # set at beginning of loop() statements.
    my @loopStack;              # current nesting of dimensions.
    my @loopCounts;             # number of dimensions in each loop.
    my @loopDims;               # dimension(s) of current loop.

    # modifiers before loop() statements.
    my $features = 0;           # bits for loop features.

    # Subst macros.
    while (my ($key, $value) = each (%macros)) {
        $codeString =~ s/\b$key\b/$value/g;
    }

    # loop thru all the tokens in the input.
    my @toks = tokenize($codeString);
    for (my $ti = 0; $ti <= $#toks; ) {
        my $tok = checkToken($toks[$ti++], '.*', 1);

        # generate simd in next loop.
        if (lc $tok eq 'simd') {

            $features |= $bSimd;
            print "info: generating SIMD in following loop.\n";
        }

        # use OpenMP on next loop.
        elsif (lc $tok eq 'omp') {

	    if ($OPT{omp} !~ /\w/) {
		warn "info: ignoring OpenMP loop modifier because '-omp' argument is empty.\n";
	    } 

	    else {
		$features |= $bOmpPar;
		print "info: using OpenMP on following loop(s).\n";
	    }
        }

        # generate manual-scheduling optimizations in next loop.
        elsif (lc $tok eq 'manual') {

            $features |= $bManual;
            print "info: using manual-scheduling optimizations.\n";
        }

        # generate nested loops.
        elsif (lc $tok eq 'nested') {

            $features |= $bNested;
            print "info: using traditional nested loops.\n";
        }

        # use tiled path in next loop if possible.
        elsif (lc $tok eq 'tiled') {
            $features |= $bTile;
        }
        
        # use serpentine path in next loop if possible.
        elsif (lc $tok eq 'serpentine') {
            $features |= $bSerp;
        }
        
        # use square_wave path in next loop if possible.
        elsif (lc $tok eq 'square_wave') {
            $features |= $bSquare;
        }
        
        # Beginning of a loop.
        # Also eats the args in parens and the following '{'.
        elsif (lc $tok eq 'loop') {

            # Get loop dimension(s).
            checkToken($toks[$ti++], '\(', 1);
            @loopDims = getArgs(\@toks, \$ti);  # might be empty.
            checkToken($toks[$ti++], '\{', 1); # eat the '{'.
            my $ndims = scalar(@loopDims);     # num dims in this loop.

            # Check index consistency.
            for my $ld (@loopDims) {
                $dims{$ld} = 1;
                $tiledDims{$ld} = 1 if ($features & $bTile);
                for my $ls (@loopStack) {
                    die "Error: loop variable '$ld' already used.\n"
                        if $ld == $ls;
                }
            }

            push @loopStack, @loopDims; # all dims so far.
            push @loopCounts, $ndims; # number of dims in each loop.
            my @loopPrefix;     # string(s) to put before loop body.

            # In inner loop?
            my $is_inner = $ndims && isInInner(\@toks, \$ti);
            push @loopPrefix, "${macroPrefix}INNER_LOOP_PREFIX" if $is_inner;

            # Add OMP pragma(s).
            my $is_omp_nested = 0;
            if ($features & $bOmpPar) {
                if (($features & $bNested) && $ndims) {
                    push @code, 
                        macroDef('OMP_NESTED_PRAGMA', undef, pragma("$OPT{omp} collapse($ndims)"));
                    push @loopPrefix, "${macroPrefix}OMP_NESTED_PRAGMA";
                    $is_omp_nested = 1;
                } else {
                    push @loopPrefix, "${macroPrefix}OMP_PRAGMA";
                }
            }
            if ($features & $bSimd) {
                push @loopPrefix, '${macroPrefix}OMP_SIMD';
            }
                
            # Start the loop unless there are no indices.
            if ($ndims) {
                print "info: generating scan over ".dimStr(@loopDims)."...\n";

                # Start the loop.
                beginLoop(\@code, \@loopDims, $features, \@loopStack, \@loopPrefix, undef, undef);

                # Inner-loop-specific code.
                if ($is_inner) {

                    # Start-stop indices for body.
                    push @code,
                        "// Indices for loop body.",
                        makeOutMacros(@loopStack),
                        "#ifdef ".outVar(),
                        "#ifndef ".inVar(),
                        "#error Cannot create ".outVar()." without ".inVar(),
                        "#endif";
                    if ($features & $bOmpPar) {
                        
                        # Make a new var so it will become OMP private.
                        push @code,
                            "ScanIndices ".outVar()."(false);",
                            outVar()." = ".inVar().";",
                            setOutVars(@loopStack);
                    } else {
                        
                        # Just a reference if no OMP.
                        push @code,
                            "ScanIndices& ".outVar()." = ".inVar().";",
                            setOutVars(@loopStack);
                    }
                    push @code, "#endif // ".outVar();
                }

            } else {

                # Dummy loop needed when there are no indices in the loop.
                # Needed to get nesting and other assumptions right.
                print "info: generating dummy loop for empty $tok args...\n";
                my $ivar = "dummy" . scalar(@loopCounts);
                push @code,
                    "// Dummy loop.",
                    @loopPrefix,
                    "for (int $ivar = 0; $ivar < 1; $ivar++) { {";
            }

            # Remove temp pragma.
            push @code, macroUndef('OMP_NESTED_PRAGMA') if $is_omp_nested;
            
            # Macro break for inserting code: end one and start next one.
            push @code,
                macroUndef($loopPart.$partNum),
                "#endif // Part $partNum.\n";
            $partNum++;
            push @code,
                "#ifdef ${macroPrefix}${loopPart}$partNum",
                "// Enable part $partNum by defining the above macro.";
            
            # clear data for this loop so we'll be ready for a nested loop.
            undef @loopDims;
            undef @loopPrefix;
            $features = 0;
        }

        # End of loop.
        elsif ($tok eq '}') {
            die "error: attempt to end loop w/o beginning\n" if !@loopCounts;

            # pop stacks.
            my $ndims = pop @loopCounts; # How many indices in this loop?
            for my $i (1..$ndims) {
                my $sdim = pop @loopStack;
                push @code, "// End of scan over dim $sdim.";
            }

            # Emit code.
            endLoop(\@code);
        }                       # end of a loop.

        # separator (ignore).
        elsif ($tok eq ';') {
        }

        # null or whitespace (ignore).
        elsif ($tok =~ /^\s*$/) {
        }

        else {
            die "error: unrecognized or unexpected token '$tok'\n";
        }
    }                           # token-handling loop.

    die "error: ".(scalar @loopStack)." loop(s) not closed.\n"
        if @loopStack;

    # Sorted list of dims scanned.
    my @dims = sort { $a <=> $b } keys %dims;
    
    # Front matter.
    my @fcode;
    push @fcode,
        "#ifdef ${macroPrefix}${loopPart}0\n",
        "// Enable part 0 by defining the above macro.",
        "// The following macros must be re-defined for each generated loop-nest.",
        macroDef('SIMD_PRAGMA', undef, pragma($OPT{simd})),
        macroDef('INNER_LOOP_PREFIX', undef, pragma($OPT{inner})),
        macroDef('OMP_PRAGMA', undef, pragma($OPT{omp})),
        macroDef('OMP_NUM_THREADS', undef, 'omp_get_num_threads()'),
        macroDef('OMP_THREAD_NUM', undef, 'omp_get_thread_num()'),
        "// Define ".inVar()." to initialize loop from a ScanIndices struct with that name.",
        "// Any element of the struct may be overridden by defining the corresponding macro.",
        "#ifdef ".inVar();
    for my $expr (@fixed_exprs) {
        push @fcode, macroDef($expr, "dim_num", inVar().'.'.$expr.'[dim_num]');
    }
    push @fcode,
        "#endif //".inVar(),
        "{",
        getInVars(\%tiledDims, @dims);
    unshift @code, @fcode;
    
    # Back matter.
    push @code,
        "}",
        macroUndef($inputVar),
        macroUndef($outputVar),
        macroUndef('OMP_PRAGMA'),
        macroUndef('OMP_NUM_THREADS'),
        macroUndef('OMP_THREAD_NUM'),
        macroUndef('SIMD_PRAGMA'),
        macroUndef('INNER_LOOP_PREFIX');
    for my $expr (@fixed_exprs, @var_exprs) {
        push @code,
            macroUndef($expr),
            macroUndef("BODY_$expr");
    }
    push @code,
        macroUndef($loopPart.$partNum),
        "#endif";
    
    # open output stream.
    open OUT, "> $OPT{output}" or die;

    # header.
    print OUT
        " /*\n",
        "  * Var-scanning code.\n",
        "  * Generated automatically from the following pseudo-code:\n",
        "  *\n",
        "  * $codeString\n",
        "  *\n */\n\n";

    # print out code.
    for my $line (@code) {
        print OUT "\n" if $line =~ m=^\s*//=; # blank line before comment.
        print OUT " $line\n"; # add space at beginning of every line.
    }
    print OUT " // End of generated var-scanning code.\n";
    close OUT;
    system("$indent $OPT{output}") if -x $indent;

    print "info: output in '$OPT{output}'.\n";
}

# Parse arguments and emit code.
sub main() {

    my(@KNOBS) = (
        # knob,        description,   optional default
        [ "prefix=s", "Common prefix of generated macros and vars", ''],
        [ "inner=s", "Set default INNER_LOOP_PREFIX macro used before inner loop(s)", ''],
        [ "omp=s", "Set default OMP_PRAGMA macro used before 'omp' loop(s)", "omp parallel for"],
        [ "simd=s", "Set default SIMD_PRAGMA macro used before 'simd' loop(s)", "omp simd"],
        [ "align!", "Generate peel code for alignment", 1],
        [ "macro_file=s", "Name of input file containing '#define' macros that can be used in <code-string>", ''],
        [ "output=s", "Name of output file", 'loops.h'],
        );
    my($command_line) = process_command_line(\%OPT, \@KNOBS);
    print "$command_line\n" if $OPT{verbose};

    my $script = basename($0);
    if (!$command_line || $OPT{help} || @ARGV < 1) {
        print "Outputs C++ code to scan N-D vars.\n",
            "Usage: $script [options] <code-string>\n",
            "The <code-string> contains optionally-nested scans across the given\n",
            "  indices indicated by 'loop(<indices>)'\n",
            "Indices may be specified as a comma-separated list or <first..last> range.\n",
            "The generated code will contain a macro-guarded part before the first loop body\n",
            "  and a part after each loop body.\n",
            "Optional loop modifiers:\n",
            "  simd:            add SIMD_PRAMA before loop (distribute work across SIMD HW).\n",
            "  omp:             add OMP_PRAGMA before loop (distribute work across SW threads).\n",
            "  nested:          use traditional nested loops instead of generating one flattened loop;\n",
            "                     automatically adds 'collapse' clause to OpenMP loops.\n",
            "  manual:          create optimized index calculation for manually-scheduled OpenMP loops;\n",
            "                     must set OMP_PRAGMA to something like 'parallel', not 'parallel for'.\n",
            "  tiled:           generate tiled scan within a >1D loop.\n",
            "  serpentine:      generate reverse scan when enclosing loop index is odd.*\n",
            "  square_wave:     generate 2D square-wave scan for two innermost dims of >1D loop.*\n",
            "      * Do not use these modifiers for YASK block or Mega-block loops because they must\n",
            "        execute with strictly-increasing indices when using temporal tiling.\n",
            "        Also, do not combile these modifiers with 'tiled' or 'manual'.\n",
            "A 'ScanIndices' type must be defined in C++ code prior to including the generated code.\n",
            "  This struct contains the following 'Indices' elements:\n",
            "  'begin':       [in] first index to scan in each dim.\n",
            "  'end':         [in] value beyond last index to scan in each dim.\n",
            "  'stride':      [in] distance between each scan point in each dim.\n",
            "  'align':       [in] alignment of strides after first one.\n",
            "  'align_ofs':   [in] value to subtract from 'start' before applying alignment.\n",
            "  'tile_size':   [in] size of each tile in each tiled dim (ignored if not tiled).\n",
            "  'start':       [out] set to first scan point in body of inner loop(s).\n",
            "  'stop':        [out] set to one past last scan point in body of inner loop(s).\n",
            "  'index':       [out] set to zero on first iteration of loop; increments each iteration.\n",
            "  Each array should be the length specified by the largest index used (typically same as -ndims).\n",
            "  The 'align' and 'align_ofs' elements are ignored if '-no-align' is used.\n",
            "A 'ScanIndices' input var must be defined in C++ code prior to including the generated code.\n",
            "  The 'in' indices control the range of the generaged loop(s).\n",
            "  The 'ScanIndices' input var is named with the LOOP_INDICES macro.\n",
            "Loop indices will be available in the body of the loop in a new 'ScanIndices' var.\n",
            "  Values in the 'out' indices are set to indicate the range to be covered by the loop body.\n",
            "  Any values in the struct not explicity set are copied from the input.\n",
            "  The 'ScanIndices' output var is named with the BODY_INDICES macro.\n",
            "Options:\n";
        print_options_help(\@KNOBS);
        print "Examples:\n",
            "  $script 'loop(0,1) { }'\n",
            "  $script 'omp loop(0,1) { }'\n",
            "  $script 'omp nested loop(0,1) { }'\n",
            "  $script 'omp serpentine loop(0,1) { }'\n",
            "  $script 'omp tiled loop(0,1,2) { }'\n",
            "  $script 'omp loop(0,1) { loop(2) { } }'\n",
            "  $script 'omp loop(0) { loop(1,2) { } }'\n",
            "  $script 'omp loop(0) { square_wave loop(1..2) { } }'\n",
            "  $script 'omp loop(0,1,2) { loop(3) { } }'\n";
        exit 1;
    }

    $macroPrefix = uc $OPT{prefix};
    $varPrefix = lc $OPT{prefix};
    $doAlign = $OPT{align};
    push @fixed_exprs, @align_exprs if $doAlign;

    # Read macros.
    if ($OPT{macro_file}) {
        open MF, "< $OPT{macro_file}" or die "cannot open '$OPT{macro_file}'\n";
        while (<MF>) {
            chomp;
            s/\s+$//;

            # Macro with name and value.
            if (/^\s*#define\s+(\w+)\s+(.*)/) {
                $macros{$1} = $2;
            }

            # Macro with name only.
            elsif (/^\s*#define\s+(\w+)/) {
                $macros{$1} = '';
            }
        }
        close MF;
        print "info: ".(scalar keys %macros)." macro(s) read from '$OPT{macro_file}'\n";
    }

    my $codeString = join(' ', @ARGV); # just concat all non-options params together.
    processCode($codeString);
}

main();
