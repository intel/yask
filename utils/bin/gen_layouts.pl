#!/usr/bin/env perl

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

# Purpose: Generate 1D<->nD layout macros.

sub usage {
  die "usage: $0 <option> <max-size>\n".
    "options:\n".
    " -p    generate perl lists of permutes\n".
    " -d    generate C++ class definitions\n".
    " -v    generate C++ var-creation code\n".
    " -m    generate CPP layout/unlayout macros\n";
}

usage() if @ARGV != 2;
my $opt = $ARGV[0];
my $max_size = $ARGV[1];
my @sizes = (1..$max_size);

use strict;
use File::Basename;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

print "// Automatically generated; do not edit.\n\n" if $opt ne '-p';

# permute items in a list.
# args: block of code to run on each permutation and list to permute.
sub permute(&@) {
  my $code = shift;

  my @idx = 0..$#_;
  while ( $code->(@_[@idx]) ) {
    my $p = $#idx;
    --$p while $idx[$p-1] > $idx[$p];
    my $q = $p or return;
    push @idx, reverse splice @idx, $p;
    ++$q while $idx[$p-1] > $idx[$q];
    @idx[$p-1,$q] = @idx[$q,$p-1];
  }
}

# generate nd->1d layout code.
sub makeLayout($$$) {
  my $a = shift;                # ref to list.
  my $jvars = shift;
  my $dvars = shift;

  my $code = '';
  for my $i (0..$#$a) {
    $code .= " + " if $i > 0;
    $code .= "$jvars->[$i]";

    # multiply by product of higher dimensions.
    map { $code .= " * $_" } @$dvars[$i+1..$#$a];
  }
  return $code;
}

# generate 1d->nd unlayout code.
sub makeUnlayout($$$$) {
  my $a = shift;                # ref to list.
  my $jvars = shift;
  my $dvars = shift;
  my $sep = shift;              # separator.

  my $code = '';
  for my $i (0..$#$a) {
    $code .= $sep if $i > 0;
    $code .= "$jvars->[$i] = (ai)";

    # divide by product of higher dimensions.
    $code .= "/(" . join(' * ', @$dvars[$i+1..$#$a]) . ")" if $i < $#$a;

    # modulo by current dimension.
    $code .= " % $dvars->[$i]" if $i > 0;
  }
  return $code;
}


# loop through number of dimensions.
for my $n (@sizes) {

  # List to be permuted.
  my @a = (1..$n);

  # macros.
  if ($opt eq '-m') {

    print "\n// $n-D <-> 1-D layout macros.\n";
    print "// 'LAYOUT' macros return 1-D offset from $n-D 'j' indices.\n".
      "// 'UNLAYOUT' macros set $n 'j' indices based on 1-D 'ai' input.\n";

    my $args = join(', ', (map { "j$_" } sort @a), (map { "d$_" } @a));

    permute {
      my @p = @_;
      my $n = join('', @p);
      my @jvars = map { "j$_" } @p;
      my @pjvars = map { "(j$_)" } @p;
      my @pdvars = map { "(d$_)" } @p;

      # n->1
      print "#define LAYOUT_$n($args) (", makeLayout(\@p, \@pjvars, \@pdvars), ")\n";

      # 1->n
      print "#define UNLAYOUT_$n(ai, $args) (", makeUnlayout(\@p, \@jvars, \@pdvars, ', '), ")\n";

    } @a;
  }

  # class decls.
  elsif ($opt eq '-d') {

    my $dvars = join(', ', map { "_d$_ = 1" } @a);
    my $cargs = join(', ', map { "idx_t d$_" } @a);
    my $cvars = join(', ', map { "d$_" } @a);
    my $cinit = join(', ', map { "_d$_(d$_)" } @a);
    my $margs = join(', ', map { "idx_t j$_" } @a);
    my $uargs = join(', ', map { "idx_t& j$_" } @a);
    my $sz = join(' * ', map { "_d$_" } @a);
    my $basename = "Layout_${n}d";

  print <<"END";

 // $n-D <-> 1-D layout base class.
 class ${basename} : public Layout {
 public:
  ${basename}() : Layout($n) { }
  ${basename}(const Indices& sizes) : Layout($n, sizes) { }
 };
END

    permute {
      my @p = @_;
      my @pm1 = map { $_ - 1; } @p;
      my $name = join('', @p);
      my @jvars = map { "j[$_]" } @pm1;
      my @dvars = map { "_sizes[$_]" } @pm1;
      my $dims = join(', ', map { "d$_" } @p);
      my $layout = makeLayout(\@p, \@jvars, \@dvars);
      my $unlayout = makeUnlayout(\@p, \@jvars, \@dvars, "; ");

      print <<"END";

 // $n-D <-> 1-D layout class with dimensions in $dims order,
 // meaning d$p[$#p] is stored with unit stride.
 class Layout_$name : public ${basename} {
 public:
  Layout_$name() { }
  Layout_$name(const Indices& sizes) : ${basename}(sizes) { }
  inline int get_num_sizes() const {
    return $n;
  }

  // Return 1-D offset from $n-D 'j' indices.
  inline idx_t layout(const Indices& j) const {
    return $layout;
  }

  // Return $n index(indices) based on 1-D 'ai' input.
  inline Indices unlayout(idx_t ai) const {
    Indices j(_sizes);
    $unlayout;
    return j;
  }
 };
END
    } @a;
  }

  # YASK ar-creation code.
  elsif ($opt eq '-v') {

    # Make type name.
    my $layout = "Layout_" . join('', 1 .. $n);

    for my $w (0 .. 1) {
      my $wrap = $w ? "true" : "false";
    
      # Creation.
      print " else if (ndims == $n && step_used == $wrap)\n",
        "  gp = make_shared<YkElemVar<$layout, $wrap>>(*this, name, gdims);\n";
    }
  }
  
  # just list permutes.
  elsif ($opt eq '-p') {

    my @strs;
    permute {
      my @p = @_;
      my $ns = join('', @p);
      push @strs, "'$ns'";
    } @a;

    print "\n# Permutations for $n dimensions.\n",
      "my \@perm$n = (", join(', ', @strs), ");\n";
  }

  # bad option.
  else {
    usage();
  }

}

