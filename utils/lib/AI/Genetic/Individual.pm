
package AI::Genetic::Individual;

use strict;
use vars qw/$VERSION/;
$VERSION = 0.02;

# this package is to serve as a base package to
# all other individuals. It doesn't do anything
# interesting.

1;

sub new {  # hmm .. do I need this?
  my ($class, $genes) = @_;

  my $self;
  if (ref $class) { # clone mode
    $self = bless {} => ref $class;
    $self->{$_} = $class->{$_} for keys %$class;
    $self->{GENES}  = $genes;
    $self->{CALCED} = 0;

  } else {          # new mode. Genome is given
    goto &newSpecific;
  }

  return $self;
}

sub new_old {  # hmm .. do I need this?
  my ($class, $genes) = @_;

  my $self;
  if (ref $class) { # clone mode
    $self = bless {} => ref $class;
    $self->{$_} = $class->{$_} for keys %$class;
    $self->{GENES}  = $genes;
    $self->{CALCED} = 0;

  } else {          # new mode. Just call newRandom.
    goto &newRandom;
  }

  return $self;
}

# should create default methods.
# those are the only three needed.
sub newRandom   {}
sub newSpecific {}
sub genes       {}

# the following methods shouldn't be overridden.
sub fitness {
  my ($self, $sub) = @_;

  $self->{FITFUNC} = $sub if $sub;
  return $self->{FITFUNC};
}

sub check {
  my ($self, $sub) = @_;

  $self->{CHECKFUNC} = $sub if $sub;
  return $self->{CHECKFUNC}->(scalar $self->genes);
}

sub score {
  my $self = shift;

  return $self->{SCORE} if $self->{CALCED};

  $self->{SCORE}  = $self->{FITFUNC}->(scalar $self->genes);
  $self->{CALCED} = 1;

  return $self->{SCORE};
}

sub resetScore { $_[0]{CALCED} = 0 }

# hmmm .. how do I reset {CALCED} in case of mutation?

__END__

=head1 NAME

AI::Genetic::Individual - Base class for AI::Genetic Individuals.

=head1 SYNOPSIS

See L<AI::Genetic>.

=head1 DESCRIPTION

This package implements the base class for all AI::Genetic individuals.
It provides basic methods required by AI::Genetic for correct evolution.
Furthermore, it can be very easily used as a base class for additional
types of individuals. AI::Genetic comes with three individual types that
inherit from this class. These are I<IndBitVector>, I<IndListVector>,
and I<IndRangeVector>.

See L</CREATING YOUR OWN INDIVIDUAL CLASS> for more details.

=head1 CLASS METHODS

The following methods are accessible publicly. They are not meant to
be over-ridden:

=over

=item I<$individual>                -E<gt>B<new(options)>

=item I<AI::Genetic::IndBitVector>  -E<gt>B<new(options)>

=item I<AI::Genetic::IndListVector> -E<gt>B<new(options)>

=item I<AI::Genetic::IndRangeVector>-E<gt>B<new(options)>

This is the default constructor. It can be called as an instance method or
as a class method. In both cases it returns a new individual of
the proper type.

If called as an instance method, it expects one argument
which defines the genes of the individual. All other attributes, like
fitness function, class, etc, will be copied from the calling
instance.

If called as a class method, then it calls I<newSpecific()>. See below
for details.

=item I<$individual>-E<gt>B<fitness(?anon_sub?)>

This method is used to set/query the anonymous subroutine used to
calculate the individual's fitness. If an argument is given, it expects
a subroutine reference which will be set as the fitness subroutine. In
either case, it'll return the fitness sub ref.

=item I<$individual>-E<gt>B<score()>

This method returns the fitness score of the individual. If the score has
never been calculated before, then the fitness function is executed and
the score saved. Subsequent calls to score() will return the cached value.

=item I<$individual>-E<gt>B<resetScore()>

This method resets the score of the individual such that a subsequent call
to I<score()> will result in the execution of the fitness sub.

=back

The following methods are meant to be over-ridden by any class that
inherits from AI::Genetic::Individual:

=over 4

=item I<$individual>-E<gt>B<newRandom(options)>

This method returns an individual with random genes. It is called with the
arguments supplied to I<AI::Genetic::init()> as explained in
L<AI::Genetic/I<$ga>-E<gt>B<init>(I<initArgs>)>.

=item I<$individual>-E<gt>B<newSpecific(options)>

This method returns an individual with the given genetic makeup. The
options depend on the type of individual:

=over

=item o bitvector

One argument is expected which is an anonymous list of genes:

  AI::Genetic::IndBitVector->new([0, 1, 1, 0, 1, 0]);

=item o listvector

Two arguments are expected. The first is an anonymous list of
genes, the second is an anonymous list of lists of possible gene values,
similar to the argument of I<newRandom>.

  AI::Genetic::IndListVector->new(
    [qw/red medium fat/],   # genes
    [  # possible values
     [qw/red blue green/],
     [qw/big medium small/],
     [qw/very_fat fat fit thin very_thin/],
    ]);

=item o rangevector

Two arguments are expected. The first is an anonymous list of
genes, the second is an anonymous list of lists of possible gene values,
similar to the argument of I<newRandom>.

  AI::Genetic::IndListVector->new(
    [3, 14, 4],   # genes
    [   # possible values
     [1, 5],
     [0, 20],
     [4, 9],
    ]);

=back

=item I<$individual>-E<gt>B<genes()>

In list context, returns a list of genes. In scalar context returns an
anonymous list of the genes.

=back

Other useful non-generic methods:

=over

=item I<$listVectorInd>-E<gt>B<lists()>

This method returns an anonymous list of lists which describes the possible
value of each gene for the given AI::Genetic::IndListVector individual. This
is the same argument passed to I<newRandom()>.

=item I<$rangeVectorInd>-E<gt>B<ranges()>

This method returns an anonymous list of lists which describes the possible
range of each gene for the given AI::Genetic::IndRangeVector individual. This
is the same argument passed to I<newRandom()>.

=back

=head1 CREATING YOUR OWN INDIVIDUAL CLASS

Creating your own individual class is easy. All you have to do is inherit from
AI::Genetic::Individual and override the I<newRandom()>, I<newSpecific>, and
I<genes()> methods to conform with the documentation above. Specifically, the
arguments to i<newRandom> and I<newSpecific> have to match what I<AI::Genetic::init()>
expects as arguments. You can also define any additional methods that you might
require in your own custom-made strategies.

Note that in order for your own individual class to be useful, you have to define
your own custom strategy that knows how to evolve such individuals. Conceptually,
this should be very simple.

=head1 AUTHOR

Written by Ala Qumsieh I<aqumsieh@cpan.org>.

=head1 COPYRIGHTS

(c) 2003,2004 Ala Qumsieh. All rights reserved.
This module is distributed under the same terms as Perl itself.

=cut

=cut

