
package AI::Genetic::IndBitVector;

use strict;
use base qw/AI::Genetic::Individual/;

1;

sub newRandom {
  my ($class, $length) = @_;

  my $self = bless {
		    GENES   => [],
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		    CALCED  => 0,
		   } => $class;

  push @{$self->{GENES}} => rand > 0.5 ? 1 : 0
    for 1 .. $length;

  return $self;
}

sub newSpecific {
  my ($class, $genes) = @_;

  my $self = bless {
		    GENES   => $genes,
		    CALCED  => 0,
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		   } => $class;

  return $self;
}

sub genes {
  my $self = shift;

  return wantarray ? @{$self->{GENES}} : [@{$self->{GENES}}];
}

