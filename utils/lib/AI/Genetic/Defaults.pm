
package AI::Genetic::Defaults;

use strict;
use AI::Genetic::OpSelection;
use AI::Genetic::OpCrossover;
use AI::Genetic::OpMutation;

1;

# this implements the default strategies.

sub rouletteSinglePoint {
  # initialize the roulette wheel
  AI::Genetic::OpSelection::initWheel($_[0]->people);

  push @_ => 'vectorSinglePoint', 'rouletteUnique';
  goto &genericStrategy;
}

sub rouletteTwoPoint {
  # initialize the roulette wheel
  AI::Genetic::OpSelection::initWheel($_[0]->people);

  push @_ => 'vectorTwoPoint', 'rouletteUnique';
  goto &genericStrategy;
}

sub rouletteUniform {
  # initialize the roulette wheel
  AI::Genetic::OpSelection::initWheel($_[0]->people);

  push @_ => 'vectorUniform', 'rouletteUnique';
  goto &genericStrategy;
}

sub tournamentSinglePoint {
  push @_ => 'vectorSinglePoint', 'tournament', [$_[0]->people];
  goto &genericStrategy;
}

sub tournamentTwoPoint {
  push @_ => 'vectorTwoPoint', 'tournament', [$_[0]->people];
  goto &genericStrategy;
}

sub tournamentUniform {
  push @_ => 'vectorUniform', 'tournament', [$_[0]->people];
  goto &genericStrategy;
}

sub randomSinglePoint {
    push @_ => 'vectorSinglePoint', 'random', [$_[0]->people];
  goto &genericStrategy;
}

sub randomTwoPoint {
  push @_ => 'vectorTwoPoint', 'random', [$_[0]->people];
  goto &genericStrategy;
}

sub randomUniform {
  push @_ => 'vectorUniform', 'random', [$_[0]->people];
  goto &genericStrategy;
}

# generic sub that implements everything.
sub genericStrategy {
  my ($ga, $Xop, $selOp, $selArgs) = @_;

  #perhaps args should be:
  # ($ga, [xop, xargs], [selop, selargs]) ?

  my $pop = $ga->people;

  # now double up the individuals, and get top half.
  my $size = $ga->size;
  my $ind  = $ga->indType;

  my @newPop;

  # optimize
  my $crossProb = $ga->crossProb;

  # figure out mutation routine to use, and its arguments.
  my @mutArgs = ($ga->mutProb);
  my $mutOp = 'bitVector';
  if      ($ind =~ /IndRangeVector/) {
    $mutOp = 'rangeVector';
    push @mutArgs => $pop->[0]->ranges;
  } elsif ($ind =~ /IndListVector/) {
    $mutOp = 'listVector';
    push @mutArgs => $pop->[0]->lists;
  }

  my ($ssub, $xsub, $msub);
  {
    no strict 'refs';
    $ssub = \&{"AI::Genetic::OpSelection::$selOp"};
    $xsub = \&{"AI::Genetic::OpCrossover::$Xop"};
    $msub = \&{"AI::Genetic::OpMutation::$mutOp"};
  }

  for my $i (1 .. $size/2) {
    my @parents = $ssub->(@$selArgs);
    @parents < 2 and push @parents => $ssub->(@$selArgs);

    my @cgenes  = $xsub->($crossProb, map scalar $_->genes, @parents);

    # check if two didn't mate.
    unless (ref $cgenes[0]) {
      @cgenes = map scalar $_->genes, @parents;
    }

    # mutate them.
    $_ = $msub->(@mutArgs, $_) for @cgenes;

    # push them into pop.
    push @newPop => map $pop->[0]->new($_), @cgenes;
  }

  # assign the fitness function. This is UGLY.
  my $fit = $pop->[0]->fitness;
  $_->fitness($fit) for @newPop;

  # now chop in half and reassign the population.
  $ga->people(AI::Genetic::OpSelection::topN([@$pop, @newPop], $size));
}
