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
  die "usage: $0 <option> <max-size>\n".
    "options:\n".
    " -p    generate perl lists of permutes\n".
    " -d    generate C++ class definitions\n".
    " -g    generate C++ grid-creation code\n".
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

if ($opt eq '-d') {
  print "#ifndef LAYOUTS_H\n".
    "#define LAYOUTS_H\n".
    "namespace yask {\n";
  print <<"END";

 // Layout base class.
 class Layout {

 protected:
  int _nsizes = 0;  // How many elements in _sizes to use (rank).
  Indices _sizes;   // Size of each dimension.

 public:
  Layout(int nsizes) : _nsizes(nsizes) {}
  Layout(int nsizes, const Indices& sizes) :
   _nsizes(nsizes), _sizes(sizes) { }
  virtual ~Layout() {}

  // Access sizes.
  const Indices& get_sizes() const { return _sizes; }
  void set_sizes(const Indices& sizes) { _sizes = sizes; }
  idx_t get_size(int i) const {
    assert(i >= 0);
    assert(i < _nsizes);
    return _sizes[i]; 
  }
  void set_size(int i, idx_t size) {
    assert(i >= 0);
    assert(i < _nsizes);
    _sizes[i] = size; 
  }
  virtual int get_num_sizes() const {
    return _nsizes; 
  }

  // Product of valid sizes.
  virtual idx_t get_num_elements() const {
    idx_t nelems = 1;
    for (int i = 0; i < _nsizes; i++)
      nelems *= _sizes[i];
    return nelems;
  }

  // Return 1-D offset from n-D 'j' indices.
  virtual idx_t layout(const Indices& j) const =0;

  // Return n indices based on 1-D 'ai' input.
  virtual Indices unlayout(idx_t ai) const =0;
 };

 // 0-D <-> 1-D layout class.
 // (Trivial layout.)
 class Layout_0d : public Layout {
 public:
  Layout_0d() : Layout(0) { }
  Layout_0d(const Indices& sizes) : Layout(0, sizes) { }
  virtual int get_num_sizes() const final {
    return 0;
  }

  // Return 1-D offset from 0-D 'j' indices.
  virtual idx_t layout(const Indices& j) const final {
    return 0; 
  }

  // Return 0 indices based on 1-D 'ai' input.
  virtual Indices unlayout(idx_t ai) const final {
    Indices j;
    return j;
  }
 };

END
}

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
  virtual int get_num_sizes() const final {
    return $n;
  }

  // Return 1-D offset from $n-D 'j' indices.
  virtual idx_t layout(const Indices& j) const final {
    return $layout;
  }

  // Return $n index(indices) based on 1-D 'ai' input.
  virtual Indices unlayout(idx_t ai) const final {
    Indices j;
    $unlayout;
    return j;
  }
 };
END
    } @a;
  }

  # grid-creation code.
  elsif ($opt eq '-g') {

    print " else if (ndims == $n) {\n";
    
    for my $fold (0 .. 1) {
      my $type = $fold ? "YkVecGrid" : "YkElemGrid";
      my $ftest = $fold ? "do_fold" : "!do_fold";
      print " if ($ftest) {\n";
      
      # Positions are 0 if they don't exist.
      # If they do,
      # - step posn is always 1st.
      # - inner posn can be anywhere.

      for my $sp (0 .. 1) {
        my $wrap = $sp ? "true" : "false";

        for my $ip (0 .. $n) {

          # can't have step and inner at same posn.
          next if $sp && $ip && $sp == $ip;

          # Make type name.
          my $layout = "Layout_";
          for my $i (1 .. $n) {

            # Add step and inner ones below.
            if ($i != $sp && $i != $ip) {
              $layout .= $i;
            }
          }

          # Step posn is always last or 2nd from last.
          if ($sp) {
            $layout .= $sp;
          }

          # Inner posn is always at end.
          if ($ip) {
            $layout .= $ip;
          }

          print " if (step_posn == $sp && inner_posn == $ip)\n",
            "  gp = make_shared<$type<$layout, $wrap>>(_dims, name, dims, &_ostr);\n";
        }
      }
      print " } // $ftest\n";
    }
    print " } // ndims == $n\n";
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

if ($opt eq '-d') {
  print "} // namespace yask.\n".
    "#endif\n";
}
