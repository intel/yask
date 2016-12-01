
package AI::Genetic::OpMutation;

use strict;

1;

# This package implements various mutation
# algorithms. To be used as static functions.

# sub bitVector():
# each gene is a bit: 0 or 1. arguments are mutation
# prob. and anon list of genes.
# returns anon list of mutated genes.

sub bitVector {
  my ($prob, $genes) = @_;

  for my $g (@$genes) {
    next if rand > $prob;

    $g = $g ? 0 : 1;
  }

  return $genes;
}

# sub rangeVector():
# each gene is a floating number, and can be anything
# within a range of two numbers.
# arguments are mutation prob., anon list of genes,
# and anon list of ranges. Each element in $ranges is
# an anon list of two numbers, min and max value of
# the corresponding gene.

sub rangeVector {
  my ($prob, $ranges, $genes) = @_;

  my $i = -1;
  for my $g (@$genes) {
    $i++;
    next if rand > $prob;

    # now randomly choose another value from the range.
    my $abs = $ranges->[$i][1] - $ranges->[$i][0] + 1;
    $g = $ranges->[$i][0] + int rand($abs);
  }

  return $genes;
}

# sub listVector():
# each gene is a string, and can be anything
# from a list of possible values supplied by user.
# arguments are mutation prob., anon list of genes,
# and anon list of value lists. Each element in $lists
# is an anon list of the possible values of
# the corresponding gene.

sub listVector {
  my ($prob, $lists, $genes) = @_;

  my $i = -1;
  for my $g (@$genes) {
    $i++;
    next if rand > $prob;

    # now randomly choose another value from the lists.
    my $new;

    if (@{$lists->[$i]} == 1) {
      $new = $lists->[$i][0];
    } else {
      do {
	$new = $lists->[$i][rand @{$lists->[$i]}];
      } while $new eq $g;
    }

    $g = $new;
  }

  return $genes;
}

__END__

=head1 NAME

AI::Genetic::OpMutation - A class that implements various mutation operators.

=head1 SYNOPSIS

See L<AI::Genetic>.

=head1 DESCRIPTION

This package implements a few mutation mechanisms that can be used in user-defined
strategies. The methods in this class are to be called as static class methods,
rather than instance methods, which means you must call them as such:

  AI::Genetic::OpCrossover::MethodName(arguments)

=head1 CLASS METHODS

There is really one kind of mutation operator implemented in this class, but it
implemented for the three default individuals types. Each gene of an individual
is looked at separately to decide whether it will be mutated or not. Mutation is
decided based upon the mutation rate (or probability). If a mutation is to happen,
then the value of the gene is switched to some other possible value.

For the case of I<bitvectors>, an ON gene switches to an OFF gene.

For the case of I<listvectors>, a gene's value is replaced by another one from
the possible list of values.

For the case of I<rangevectors>, a gene's value is replaced by another one from
the possible range of integers.

Thus, there are only three methods:

=over

=item B<bitVector>(I<mut_prob, genes>)

The method takes as input the mutation rate, and an anonymous list of genes of
a bitvector individual. The return value is an anonymous list of mutated genes.
Note that
it is possible that no mutation will occur, and thus the returned genes are
identical to the given ones.

=item B<listVector>(I<mut_prob, genes, possibleValues>)

The method takes as input the mutation rate, an anonymous list of genes of
a listvector individual, and a list of lists which describe the possible
values for each gene. The return value is an anonymous list of mutated genes.
Note that
it is possible that no mutation will occur, and thus the returned genes are
identical to the given ones.

=item B<rangeVector>(I<mut_prob, genes, rangeValues>)

The method takes as input the mutation rate, an anonymous list of genes of
a rangevector individual, and a list of lists which describe the range of 
possible values for each gene. The return value is an anonymous list of
mutated genes. Note that
it is possible that no mutation will occur, and thus the returned genes are
identical to the given ones.

=back

=head1 AUTHOR

Written by Ala Qumsieh I<aqumsieh@cpan.org>.

=head1 COPYRIGHTS

(c) 2003,2004 Ala Qumsieh. All rights reserved.
This module is distributed under the same terms as Perl itself.

=cut

