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

# Purpose: Generate 1D<->nD layout macros.

sub usage {
  die "usage: $0 <option>\n".
    " -p    generate perl lists of permutes\n".
    " -d    generate C++ class definitions\n".
    " -c    generate macro-based C++ class definitions (deprecated)\n".
    " -m    generate CPP layout/unlayout macros\n";
}

usage() if !defined $ARGV[0];
my $opt = $ARGV[0];
my @sizes = (1..4);

use strict;
use File::Basename;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

# If this module is not installed, download from
# http://search.cpan.org/~tyemq/Algorithm-Loops-1.031/lib/Algorithm/Loops.pm
use Algorithm::Loops qw(NextPermuteNum);

print "// Automatically generated; do not edit.\n";
print "#include <stddef.h>\n" if ($opt eq '-d');

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

  # argument for permute().
  my @a = (1..$n);

  # print out old matrix classes.
  if ($opt eq '-c') {

    # RealMatrix only defined for 4D.
    next if $n != 4;

    print "\n// Matrix declarations for $n dimensions.\n";

    do {
      my $ns = join('', @a);
      print "#define RMAT_CLASS RealMatrix$ns\n",
        "#define RMAT_LAYOUT LAYOUT$ns\n",
        "#include \"real_matrix.hpp\"\n"; }
      while (NextPermuteNum @a);
  }

  # macros.
  elsif ($opt eq '-m') {

    print "\n// $n-D <-> 1-D layout macros.\n";
    print "// 'LAYOUT' macros return 1-D offset from $n-D 'j' indices.\n".
      "// 'UNLAYOUT' macros set $n 'j' indices based on 1-D 'ai' input.\n";

    my $args = join(', ', (map { "j$_" } sort @a), (map { "d$_" } @a));

    do {
      my $n = join('', @a);
      my @jvars = map { "j$_" } @a;
      my @pjvars = map { "(j$_)" } @a;
      my @pdvars = map { "(d$_)" } @a;

      # n->1
      print "#define LAYOUT_$n($args) (", makeLayout(\@a, \@pjvars, \@pdvars), ")\n";

      # 1->n
      print "#define UNLAYOUT_$n(ai, $args) (", makeUnlayout(\@a, \@jvars, \@pdvars, ', '), ")\n";

    } while (NextPermuteNum @a);
  }

  # class decls.
  elsif ($opt eq '-d') {

    my $vars = join(', ', map { "_d$_" } @a);
    my $cargs = join(', ', map { "idx_t d$_" } @a);
    my $cvars = join(', ', map { "d$_" } @a);
    my $cinit = join(', ', map { "_d$_(d$_)" } @a);
    my $margs = join(', ', map { "idx_t j$_" } @a);
    my $uargs = join(', ', map { "idx_t& j$_" } @a);
    my $sz = join(' * ', map { "_d$_" } @a);
    my $basename = "Layout_${n}d";

    print "\n// $n-D <-> 1-D layout base class.\n",
      "class ${basename} {\n",
      "protected:\n",
      "  idx_t $vars;\n\n",
      "public:\n\n",
      "  ${basename}($cargs) : $cinit { }\n\n";
    for my $a (@a) {
      print "  // Return dimension $a.\n",
        "  virtual idx_t get_d$a() const { return _d$a; };\n\n";
    }
    print "  // Return overall number of elements.\n",
      "  virtual idx_t get_size() const { return $sz; };\n\n",
      "  // Return 1-D offset from $n-D 'j' indices.\n",
      "  virtual idx_t layout($margs) const =0;\n\n",
      "  // Set $n 'j' indices based on 1-D 'ai' input.\n",
      "  virtual void unlayout(idx_t ai, $uargs) const =0;\n",
      "};\n";

    do {
      my $name = join('', @a);
      my @jvars = map { "j$_" } @a;
      my @dvars = map { "_d$_" } @a;
      my $dims = join(', ', map { "d$_" } @a);

      print "\n// $n-D <-> 1-D layout class with dimensions in $dims order,\n",
        "// meaning d$a[$#a] is stored with unit stride.\n",
        "class Layout_$name : public ${basename} {\n",
        "public:\n\n",
        "  Layout_$name($cargs) : ${basename}($cvars) { }\n\n",
        "  // Return 1-D offset from $n-D 'j' indices.\n",
        "  virtual idx_t layout($margs) const\n",
        "    { return ", makeLayout(\@a, \@jvars, \@dvars), "; }\n\n",
        "  // set $n 'j' indices based on 1-D 'ai' input.\n",
        "  virtual void unlayout(idx_t ai, $uargs) const\n",
        "    { ", makeUnlayout(\@a, \@jvars, \@dvars, "; "), "; }\n",
        "};\n";

    } while (NextPermuteNum @a);
  }

  # just list permutes.
  elsif ($opt eq '-p') {

    print "# Permutations for $n dimensions.\n";
    print "(";

    do {
      my $ns = join('', @a);
      print "'$ns', ";
    } while (NextPermuteNum @a);

    print ");\n";
  }

  # bad option.
  else {
    usage();
  }

}
