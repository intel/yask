
package AI::Genetic::IndRangeVector;

use strict;
use base qw/AI::Genetic::Individual/;

1;

sub newRandom {
  my ($class, $ranges) = @_;

  my $self = bless {
		    GENES   => [],
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		    CALCED  => 0,
		    RANGES  => $ranges,
		   } => $class;

  for my $r (@$ranges) {
    my $rand = $r->[0] + int rand($r->[1] - $r->[0] + 1);
    push @{$self->{GENES}} => $rand;
  }

  return $self;
}

sub newSpecific {
  my ($class, $genes, $ranges) = @_;

  my $self = bless {
		    GENES   => $genes,
		    CALCED  => 0,
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		    RANGES  => $ranges,
		   } => $class;

  return $self;
}

sub genes {
  my $self = shift;

  return wantarray ? @{$self->{GENES}} : [@{$self->{GENES}}];
}

sub ranges { $_[0]{RANGES} }
