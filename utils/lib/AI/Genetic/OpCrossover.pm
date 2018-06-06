
package AI::Genetic::OpCrossover;

use strict;

1;

# sub vectorSinglePoint():
# Single point crossover.
# arguments are crossover prob, two
# anon lists of genes (parents).
# If crossover occurs, returns two anon lists
# of children genes. If no crossover, returns 0.
# both parents have to be of same length.

sub vectorSinglePoint {
  my ($prob, $mom, $dad) = @_;

  return 0 if rand > $prob;

  # get single index from 1 to $#{$dad}
  my $ind = 1 + int rand $#{$dad};

  my @c1 = (@$mom[0 .. $ind - 1],
	    @$dad[$ind .. $#{$dad}]);
  my @c2 = (@$dad[0 .. $ind - 1],
	    @$mom[$ind .. $#{$dad}]);

  return (\@c1, \@c2);
}

# sub vectorTwoPoint():
# Two point crossover.
# arguments are crossover prob, two
# anon lists of genes (parents).
# If crossover occurs, returns two anon lists
# of children genes. If no crossover, returns 0.
# both parents have to be of same length.

sub vectorTwoPoint {
  my ($prob, $mom, $dad) = @_;

  return 0 if rand > $prob;

  # get first index from 1 to $#{$dad}-1
  my $ind1 = 1 + int rand($#{$dad} - 1);

  # get second index from $ind1 to $#{$dad}
  my $ind2 = $ind1 + 1 + int rand($#{$dad} - $ind1);
  my @c1 = (@$mom[0 .. $ind1 - 1],
	    @$dad[$ind1 .. $ind2 - 1],
	    @$mom[$ind2 .. $#{$dad}]);

  my @c2 = (@$dad[0 .. $ind1 - 1],
	    @$mom[$ind1 .. $ind2 - 1],
	    @$dad[$ind2 .. $#{$dad}]);

  return (\@c1, \@c2);
}

# sub vectorUniform():
# Uniform crossover.
# arguments are crossover prob, two
# anon lists of genes (parents).
# If crossover occurs, returns two anon lists
# of children genes. If no crossover, returns 0.
# both parents have to be of same length.

sub vectorUniform {
  my ($prob, $mom, $dad) = @_;

  return 0 if rand > $prob;

  my (@c1, @c2);
  for my $i (0 .. $#{$dad}) {
    if (rand > 0.5) {
      push @c1 => $mom->[$i];
      push @c2 => $dad->[$i];
    } else {
      push @c2 => $mom->[$i];
      push @c1 => $dad->[$i];
    }
  }

  return (\@c1, \@c2);
}

__END__

=head1 NAME

AI::Genetic::OpCrossover - A class that implements various crossover operators.

=head1 SYNOPSIS

See L<AI::Genetic>.

=head1 DESCRIPTION

This package implements a few crossover mechanisms that can be used in user-defined
strategies. The methods in this class are to be called as static class methods,
rather than instance methods, which means you must call them as such:

  AI::Genetic::OpCrossover::MethodName(arguments)

=head1 CROSSOVER OPERATORS AND THEIR METHODS

The following crossover operators are defined:

=over

=item Single Point

In single point crossover, a point is selected along the choromosomes of both parents.
The chromosomes are then split at that point, and the head of one parent chromosome is
joined with the tail of the other and vice versa, creating two child chromosomes.
The following method is defined:

=over

=item B<vectorSinglePoint>(I<Xprob, parent1, parent2>)

The first argument is the crossover rate. The second and third arguments are anonymous
lists that define the B<genes>
of the parents (not AI::Genetic::Individual objects, but the return value of the I<genes()>
method in scalar context).
If mating occurs, two anonymous lists of genes are returned corresponding to the two
new children. If no mating occurs, 0 is returned.

=back

=item Two Point

In two point crossover, two points are selected along the choromosomes of both parents.
The chromosomes are then cut at those points, and the middle parts are swapped,
creating two child chromosomes. The following method is defined:

=over

=item B<vectorTwoPoint>(I<Xprob, parent1, parent2>)

The first argument is the crossover rate. The second and third arguments are anonymous
lists that define the B<genes>
of the parents (not AI::Genetic::Individual objects, but the return value of the I<genes()>
method in scalar context).
If mating occurs, two anonymous lists of genes are returned corresponding to the two
new children. If no mating occurs, 0 is returned.

=back

=item Uniform

In uniform crossover, two child chromosomes are created by looking at each gene in both
parents, and randomly selecting which one to go with each child.
The following method is defined:

=over

=item B<vectorUniform>(I<Xprob, parent1, parent2>)

The first argument is the crossover rate. The second and third arguments are anonymous
lists that define the B<genes>
of the parents (not AI::Genetic::Individual objects, but the return value of the I<genes()>
method in scalar context).
If mating occurs, two anonymous lists of genes are returned corresponding to the two
new children. If no mating occurs, 0 is returned.

=back

=back

=head1 AUTHOR

Written by Ala Qumsieh I<aqumsieh@cpan.org>.

=head1 COPYRIGHTS

(c) 2003,2004 Ala Qumsieh. All rights reserved.
This module is distributed under the same terms as Perl itself.

=cut

