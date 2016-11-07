
package AI::Genetic::OpSelection;

use strict;

my @wheel;
my $wheelPop;

# sub init():
# initializes the roulette wheel array.
# must be called whenever the population changes.
# only useful for roulette().

sub initWheel {
  my $pop = shift;

  my $tot = 0;
  for (@$pop) {
    my $s = $_->score;
    if ($s < 0) {
      warn "warning: scores must be >= 0; changing $s to 0\n";
      $s = 0;
    }
    $tot += $s;
  }

  # normalize
  # fix if $tot==0;
  my @norms = map { $tot ? $_->score / $tot : 1/scalar(@$pop) } @$pop;

  @wheel = ();

  my $cur = 0;
  for my $i (@norms) {
    push @wheel => [$cur, $cur + $i];
    $cur += $i;
  }

  $wheelPop = $pop;
}

# sub roulette():
# Roulette Wheel selection.
# argument is number of individuals to select (def = 2).
# returns selected individuals.

sub roulette {
  my $num = shift || 2;

  my @selected;

  for my $j (1 .. $num) {
    my $rand = rand;
    for my $i (0 .. $#wheel) {
      if ($wheel[$i][0] <= $rand && $rand < $wheel[$i][1]) {
	push @selected => $wheelPop->[$i];
	last;
      }
    }
  }

  return @selected;
}

# same as roulette(), but returns unique individuals.
sub rouletteUnique {
  my $num = shift || 2;

  # make sure we select unique individuals.
  my %selected;

  my $tries = 0;
  while ($num > keys %selected) {
    my $rand = rand;

    # if it's too hard to find using weighted wheel, just pick evenly.
    if ($tries++ > $num * 1000) {
      my $i = int rand($#wheel+1);
      $selected{$i} = 1;
    }

    else {
      for my $i (0 .. $#wheel) {
        if ($wheel[$i][0] <= $rand && $rand < $wheel[$i][1]) {
          $selected{$i} = 1;
          last;
        }
      }
    }
  }

  return map $wheelPop->[$_], keys %selected;
}

# sub tournament():
# arguments are anon list of population, and number
# of individuals in tournament (def = 2).
# return 1 individual.

sub tournament {
  my ($pop, $num) = @_;

  $num ||= 2;

  my %s;
  while ($num > keys %s) {
    my $i = int rand @$pop;
    $s{$i} = 1;
  }

  # Force eval of score to keep from breaking $_.
  my @p = map $pop->[$_], keys %s;
  for my $i (@p) {
    $i->score;
  }
  
  return (sort {$b->{SCORE} <=> $a->{SCORE}} @p)[0];
}

# sub random():
# pure random choice of individuals.
# arguments are anon list of population, and number
# of individuals to select (def = 1).
# returns selected individual(s).

sub random {
  my ($pop, $num) = @_;

  $num ||= 1;

  my %s;
  while ($num > keys %s) {
    my $i = int rand @$pop;
    $s{$i} = 1;
  }

  return map $pop->[$_], keys %s;
}

# sub topN():
# fittest N individuals.
# arguments are anon list of pop, and N (def = 1).
# return anon list of top N individuals.

sub topN {
  my ($pop, $N) = @_;

  $N ||= 1;

  # Force eval to avoid breaking $_.
  for my $i (@$pop) {
    $i->score;
  }

  # hmm .. are inputs already sorted?
  return [(sort {
    die if !defined $a;
    die if !defined $b;
    $b->{SCORE} <=> $a->{SCORE};
  } @$pop)[0 .. $N-1]];
}

1;

__END__

=head1 NAME

AI::Genetic::OpSelection - A class that implements various selection operators.

=head1 SYNOPSIS

See L<AI::Genetic>.

=head1 DESCRIPTION

This package implements a few selection mechanisms that can be used in user-defined
strategies. The methods in this class are to be called as static class methods,
rather than instance methods, which means you must call them as such:

  AI::Genetic::OpSelection::MethodName(arguments)

=head1 SELECTION OPERATORS AND THEIR METHODS

The following selection operators are defined:

=over

=item Roulette Wheel

Here, the probability of an individual being selected is proportional to its fitness
score. The following methods are defined:

=over

=item B<initWheel>(I<population>)

This method initializes the roulette wheel. It expects an anonymous list of individuals,
as returned by the I<people()> method as described in L<AI::Genetic/CLASS METHODS>.
This B<must> be called only once.

=item B<roulette>(?I<N>?)

This method selects I<N> individuals. I<N> defaults to 2. Note that the same individual
can be selected multiple times.

=item B<rouletteUnique>(?I<N>?)

This method selects I<N> unique individuals. I<N> defaults to 2. Any individual
can be selected only once per call to this method.

=back

=item Tournament

Here, I<N> individuals are randomly selected, and the fittest one of
them is returned. The following method is defined:

=over

=item B<tournament>(?I<N>?)

I<N> defaults to 2. Note that only one individual is returned per call to this
method.

=back

=item Random

Here, I<N> individuals are randomly selected and returned.
The following method is defined:

=over

=item B<random>(?I<N>?)

I<N> defaults to 1.

=back

=item Fittest

Here, the fittest I<N> individuals of the whole population are returned.
The following method is defined:

=over

=item B<topN>(?I<N>?)

I<N> defaults to 1.

=back

=back

=head1 AUTHOR

Written by Ala Qumsieh I<aqumsieh@cpan.org>.

=head1 COPYRIGHTS

(c) 2003,2004 Ala Qumsieh. All rights reserved.
This module is distributed under the same terms as Perl itself.

=cut

