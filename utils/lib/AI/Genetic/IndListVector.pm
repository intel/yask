
package AI::Genetic::IndListVector;

use strict;
use base qw/AI::Genetic::Individual/;

1;

sub newRandom {
  my ($class, $lists) = @_;

  my $self = bless {
		    GENES   => [],
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		    CALCED  => 0,
		    LISTS   => $lists,
		   } => $class;

  push @{$self->{GENES}} => $_->[rand @$_] for @$lists;

  return $self;
}

sub newSpecific {
  my ($class, $genes, $lists) = @_;

  my $self = bless {
		    GENES   => $genes,
		    CALCED  => 0,
		    SCORE   => 0,
		    FITFUNC => sub {},
		    CHECKFUNC => sub {},
		    LISTS   => $lists,
		   } => $class;

  return $self;
}

sub genes {
  my $self = shift;

  return wantarray ? @{$self->{GENES}} : [@{$self->{GENES}}];
}

sub lists { $_[0]{LISTS} }
