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

# Purpose: Generate 1D<->nD mapping macros.

sub usage {
  die "usage: $0 <option>\n".
    " -p    generate perl lists of permutes\n".
    " -c    generate C++ class definitions\n".
    " -m    generate CPP map/unmap macros\n";
}

usage() if !defined $ARGV[0];
my $opt = $ARGV[0];
my @sizes = (2..4);

use strict;
use File::Basename;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

# If this module is not installed, download from
# http://search.cpan.org/~tyemq/Algorithm-Loops-1.031/lib/Algorithm/Loops.pm
use Algorithm::Loops qw(NextPermuteNum);

print "// Automatically generated; do not edit.\n";

# loop through number of dimensions.
for my $n (@sizes) {

  # argument for permute().
  my @a = (1..$n);

  # print out matrix classes.
  if ($opt eq '-c') {

    # RealMatrix only defined for 4D.
    next if $n != 4;

    print "\n// Matrix declarations for $n dimensions.\n";

    do {
      my $ns = join('', @a);
      print "#define RMAT_CLASS RealMatrix$ns\n",
        "#define RMAT_MAP MAP$ns\n",
        "#include \"real_matrix.hpp\"\n"; }
      while (NextPermuteNum @a);
  }

  # macros.
  elsif ($opt eq '-m') {

    print "\n// $n-D <-> 1-D mapping macros.\n";
    print "// 'MAP' macros return 1-D offset from $n-D 'j' indices.\n".
      "// 'UNMAP' macros set $n 'j' indices based on 1-D 'ai' input.\n";

    do {
      my @jvars = map { "j$_" } @a;
      my @dvars = map { "d$_" } @a;
      my @pjvars = map { "(j$_)" } @a;
      my @pdvars = map { "(d$_)" } @a;
      my $n = join('', @a);
      my $args = join(', ', (map { "j$_" } sort @a), (map { "d$_" } sort @a));

      # n->1
      print "#define MAP$n($args) ";

      # calculation.
      print "(";
      for my $i (0..$#a) {
        print " + " if $i > 0;
        print "$pjvars[$i]";

        # multiply by product of higher dimensions.
        print map { "*$_" } @pdvars[$i+1..$#a];
      }
      print ")\n";

      # 1->n
      print "#define UNMAP$n(ai, $args) ";

      # calculation.
      print "(";
      for my $i (0..$#a) {
        print ", " if $i > 0;
        print "$jvars[$i] = (ai)";

        # divide by product of higher dimensions.
        print "/(", join('*', @pdvars[$i+1..$#a]), ")" if $i < $#a;

        # modulo.
        print "%$pdvars[$i]" if $i > 0;
      }
      print ")\n";


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
