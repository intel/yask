#!/usr/bin/env perl

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

# Purpose: Generate 1D<->nD layout macros.

sub usage {
  die "usage: $0 <option>\n".
    " -p    generate perl lists of permutes\n".
    " -d    generate C++ class definitions\n".
    " -m    generate CPP layout/unlayout macros\n";
}

usage() if !defined $ARGV[0];
my $opt = $ARGV[0];
my @sizes = (1..5);

use strict;
use File::Basename;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

print "// Automatically generated; do not edit.\n";
print "#include <stddef.h>\n" if ($opt eq '-d');

# permute items in a list.
# args: block of code and a list.
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

    permute {
      my @p = @_;
      my $name = join('', @p);
      my @jvars = map { "j$_" } @p;
      my @dvars = map { "_d$_" } @p;
      my $dims = join(', ', map { "d$_" } @p);

      print "\n// $n-D <-> 1-D layout class with dimensions in $dims order,\n",
        "// meaning d$p[$#p] is stored with unit stride.\n",
        "class Layout_$name : public ${basename} {\n",
        "public:\n\n",
        "  Layout_$name($cargs) : ${basename}($cvars) { }\n\n",
        "  // Return 1-D offset from $n-D 'j' indices.\n",
        "  virtual idx_t layout($margs) const\n",
        "    { return ", makeLayout(\@p, \@jvars, \@dvars), "; }\n\n",
        "  // set $n 'j' indices based on 1-D 'ai' input.\n",
        "  virtual void unlayout(idx_t ai, $uargs) const\n",
        "    { ", makeUnlayout(\@p, \@jvars, \@dvars, "; "), "; }\n",
        "};\n";

    } @a;
  }

  # just list permutes.
  elsif ($opt eq '-p') {

    print "# Permutations for $n dimensions.\n";

    for my $i (0..1) {
      my @b;
      if ($i == 0) {
        @b = @a;
      } else {
        @b = map { my $b = $_;
                   if (@a == 4) {
                     $b =~ tr/1234/nxyz/;
                   } elsif (@a == 3) {
                     $b =~ tr/123/xyz/;
                   } elsif (@a == 2) {
                     $b =~ tr/12/xy/;
                   } else {
                     $b =~ tr/1/x/;
                   }
                   $b } @a;
      }
      
      print "(";
      permute {
        my @p = @_;
        my $ns = join('', @p);
        print "'$ns', ";
      } @b;
      print ")\n";
    }
  }

  # bad option.
  else {
    usage();
  }

}
