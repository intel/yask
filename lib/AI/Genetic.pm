
package AI::Genetic;

use strict;
use Carp;

use vars qw/$VERSION/;

$VERSION = 0.05;

use AI::Genetic::Defaults;

# new AI::Genetic. More modular.
# Not too many checks are done still.

##### Shared private vars
# this hash predefines some strategies

my %_strategy = (
		 rouletteSinglePoint => \&AI::Genetic::Defaults::rouletteSinglePoint,
		 rouletteTwoPoint    => \&AI::Genetic::Defaults::rouletteTwoPoint,
		 rouletteUniform     => \&AI::Genetic::Defaults::rouletteUniform,

		 tournamentSinglePoint => \&AI::Genetic::Defaults::tournamentSinglePoint,
		 tournamentTwoPoint    => \&AI::Genetic::Defaults::tournamentTwoPoint,
		 tournamentUniform     => \&AI::Genetic::Defaults::tournamentUniform,

		 randomSinglePoint => \&AI::Genetic::Defaults::randomSinglePoint,
		 randomTwoPoint    => \&AI::Genetic::Defaults::randomTwoPoint,
		 randomUniform     => \&AI::Genetic::Defaults::randomUniform,
		);

# this hash maps the genome types to the
# classes they're defined in.

my %_genome2class = (
		     bitvector   => 'AI::Genetic::IndBitVector',
		     rangevector => 'AI::Genetic::IndRangeVector',
		     listvector  => 'AI::Genetic::IndListVector',
		    );

##################

# sub new():
# This is the constructor. It creates a new AI::Genetic
# object. Options are:
# -population: set the population size
# -crossover:  set the crossover probability
# -mutation:   set the mutation probability
# -fitness:    set the fitness function
# -check:      set the initial population viability-check function
# -type:       set the genome type. See docs.
# -terminate:  set termination sub.

sub new {
  my ($class, %args) = @_;

  my $self = bless {
		    ADDSEL => {},   # user-defined selections
		    ADDCRS => {},   # user-defined crossovers
		    ADDMUT => {},   # user-defined mutations
		    ADDSTR => {},   # user-defined strategies
		   } => $class;

  $self->{FITFUNC}    = $args{-fitness}    || sub { 1 }; # default returns '1'.
  $self->{CHECKFUNC}  = $args{-check}      || sub { 1 }; # default always passes.
  $self->{CROSSRATE}  = $args{-crossover}  || 0.95;
  $self->{MUTPROB}    = $args{-mutation}   || 0.05;
  $self->{POPSIZE}    = $args{-population} || 100;
  $self->{TYPE}       = $args{-type}       || 'bitvector';
  $self->{TERM}       = $args{-terminate}  || sub { 0 };
  $self->{MAXCHECKFAILS}  = $args{-maxCheckFails}  || 1000; # max tries to make an individual.

  $self->{PEOPLE}     = [];   # list of individuals
  $self->{GENERATION} = 0;    # current gen.

  $self->{INIT}       = 0;    # whether pop is initialized or not.
  $self->{SORTED}     = 0;    # whether the population is sorted by score or not.
  $self->{INDIVIDUAL} = '';   # name of individual class to use().

  return $self;
}

# sub createStrategy():
# This method creates a new strategy.
# It takes two arguments: name of strategy, and
# anon sub that implements it.

sub createStrategy {
  my ($self, $name, $sub) = @_;

  if (ref($sub) eq 'CODE') {
    $self->{ADDSTR}{$name} = $sub;
  } else {
    # we don't know what this operation is.
    carp <<EOC;
ERROR: Must specify anonymous subroutine for strategy.
       Strategy '$name' will be deleted.
EOC
    ;
    delete $self->{ADDSTR}{$name};
    return undef;
  }

  return $name;
}

# sub evolve():
# This method evolves the population using a specific strategy
# for a specific number of generations.

sub evolve {
  my ($self, $strategy, $gens) = @_;

  unless ($self->{INIT}) {
    carp "can't evolve() before init()";
    return undef;
  }

  my $strSub;
  if      (exists $self->{ADDSTR}{$strategy}) {
    $strSub = $self->{ADDSTR}{$strategy};
  } elsif (exists $_strategy{$strategy}) {
    $strSub = $_strategy{$strategy};
  } else {
    carp "ERROR: Do not know what strategy '$strategy' is,";
    return undef;
  }

  $gens ||= 1;

  for my $i (1 .. $gens) {
    $self->sortPopulation;
    $strSub->($self);

    $self->{GENERATION}++;
    $self->{SORTED} = 0;

    last if $self->{TERM}->($self);

#    my @f = $self->getFittest(10);
#    for my $f (@f) {
#      print STDERR "    Fitness = ", $f->score, "..\n";
#      print STDERR "    Genes are: @{$f->genes}.\n";
#    }
  }
}

# sub sortIndividuals():
# This method takes as input an anon list of individuals, and returns
# another anon list of the same individuals but sorted in decreasing
# score.

sub sortIndividuals {
  my ($self, $list) = @_;

  # make sure all scores are calculated.
  # This is to avoid a bug in Perl where a sort is called from within another
  # sort, and they are in different packages, then you get a use of uninit value
  # warning. See http://rt.perl.org/rt3/Ticket/Display.html?id=7063
  for my $i (@$list) {
    $i->score;
  }

  return [sort {
    die if !defined $b;
    die if !defined $a;
    $b->{SCORE} <=> $a->{SCORE};
  } @$list];
}

# sub sortPopulation():
# This method sorts the population of individuals.

sub sortPopulation {
  my $self = shift;

  return if $self->{SORTED};

  $self->{PEOPLE} = $self->sortIndividuals($self->{PEOPLE});
  $self->{SORTED} = 1;
}

# sub getFittest():
# This method returns the fittest individuals.

sub getFittest {
  my ($self, $N) = @_;

  $N ||= 1;
  $N = 1 if $N < 1;

  $N = @{$self->{PEOPLE}} if $N > @{$self->{PEOPLE}};

  $self->sortPopulation;

  my @r = @{$self->{PEOPLE}}[0 .. $N-1];

  return $r[0] if $N == 1 && not wantarray;

  return @r;
}

# sub init():
# This method initializes the population to completely
# random individuals. It deletes all current individuals!!!
# It also examines the type of individuals we want, and
# require()s the proper class. Throws an error if it can't.
# Must pass to it an anon list that will be passed to the
# newRandom method of the individual.

# In case of bitvector, $newArgs is length of bitvector.
# In case of rangevector, $newArgs is anon list of anon lists.
# each sub-anon list has two elements, min number and max number.
# In case of listvector, $newArgs is anon list of anon lists.
# Each sub-anon list contains possible values of gene.

sub init {
  my ($self, $newArgs) = @_;

  $self->{INIT} = 0;

  my $ind;
  if (exists $_genome2class{$self->{TYPE}}) {
    $ind = $_genome2class{$self->{TYPE}};
  } else {
    $ind = $self->{TYPE};
  }

  eval "use $ind";  # does this work if package is in same file?
  if ($@) {
    carp "ERROR: Init failed. Can't require '$ind': $@";
    return undef;
  }

  $self->{INDIVIDUAL} = $ind;
  $self->{PEOPLE}     = [];
  $self->{SORTED}     = 0;
  $self->{GENERATION} = 0;
  $self->{INITARGS}   = $newArgs;

  for my $i (1 .. $self->{POPSIZE}) {

    # keep trying until check passes or limit reached.
    my $ok = 0;
    for my $tries (1 .. $self->{MAXCHECKFAILS}) {
      my $n = $ind->newRandom($newArgs);
      if ($n->check($self->{CHECKFUNC})) {
        push @{$self->{PEOPLE}} => $n;
        $ok = 1;
        last;
      }
    }
    if (!$ok) {
      carp "ERROR: unable to create a viable individual in $self->{MAXCHECKFAILS} tries";
      return undef;
    }
  }

  for my $i (@{$self->{PEOPLE}}) {
    $i->fitness($self->{FITFUNC});
  }

  $self->{INIT} = 1;
}

# sub people():
# returns the current list of individuals in the population.
# note: this returns the actual array ref, so any changes
# made to it (ex, shift/pop/etc) will be reflected in the
# population.

sub people {
  my $self = shift;

  if (@_) {
    $self->{PEOPLE} = shift;
    $self->{SORTED} = 0;
  }

  $self->{PEOPLE};
}

# useful little methods to set/query parameters.
sub size       { $_[0]{POPSIZE}    = $_[1] if defined $_[1]; $_[0]{POPSIZE}   }
sub crossProb  { $_[0]{CROSSRATE}  = $_[1] if defined $_[1]; $_[0]{CROSSRATE} }
sub mutProb    { $_[0]{MUTPROB}    = $_[1] if defined $_[1]; $_[0]{MUTPROB}   }
sub indType    { $_[0]{INDIVIDUAL} }
sub generation { $_[0]{GENERATION} }

# sub inject():
# This method is used to add individuals to the current population.
# The point of it is that sometimes the population gets stagnant,
# so it could be useful add "fresh blood".
# Takes a variable number of arguments. The first argument is the
# total number, N, of new individuals to add. The remaining arguments
# are genomes to inject. There must be at most N genomes to inject.
# If the number, n, of genomes to inject is less than N, N - n random
# genomes are added. Perhaps an example will help?
# returns 1 on success and undef on error.

sub inject {
  my ($self, $count, @genomes) = @_;

  unless ($self->{INIT}) {
    carp "can't inject() before init()";
    return undef;
  }

  my $ind = $self->{INDIVIDUAL};

  my @newInds;
  for my $i (1 .. $count) {
    my $genes = shift @genomes;

    if ($genes) {
      push @newInds => $ind->newSpecific($genes, $self->{INITARGS});
    } else {
      push @newInds => $ind->newRandom  ($self->{INITARGS});      
    }
  }

  $_->fitness($self->{FITFUNC}) for @newInds;

  push @{$self->{PEOPLE}} => @newInds;

  return 1;
}

__END__

=head1 NAME

AI::Genetic - A pure Perl genetic algorithm implementation.

=head1 SYNOPSIS

    use AI::Genetic;
    my $ga = new AI::Genetic(
        -fitness    => \&fitnessFunc,
        -check      => \&checkFunc,
        -type       => 'bitvector',
        -population => 500,
        -crossover  => 0.9,
        -mutation   => 0.01,
	-terminate  => \&terminateFunc,
       );

     $ga->init(10);
     $ga->evolve('rouletteTwoPoint', 100);
     print "Best score = ", $ga->getFittest->score, ".\n";

     sub checkFunc {
         my $genes = shift;

         my $ok;
         # determine whether $genes is probably ok.
         
         return $ok;
     }

     sub fitnessFunc {
         my $genes = shift;

         my $fitness;
         # assign a number to $fitness based on the @$genes
         # ...

         return $fitness;
      }

      sub terminateFunc {
         my $ga = shift;

         # terminate if reached some threshold.
         return 1 if $ga->getFittest->score > $THRESHOLD;
         return 0;
      }

=head1 DESCRIPTION

This module implements a Genetic Algorithm (GA) in pure Perl.
Other Perl modules that achieve the same thing (perhaps better,
perhaps worse) do exist. Please check CPAN. I mainly wrote this
module to satisfy my own needs, and to learn something about GAs
along the way.

B<PLEASE NOTE:> As of v0.02, AI::Genetic has been re-written from
scratch to be more modular and expandable. To achieve this, I had
to modify the API, so it is not backward-compatible with v0.01.
As a result, I do not plan on supporting v0.01.

I will not go into the details of GAs here, but here are the
bare basics. Plenty of information can be found on the web.

In a GA, a population of individuals compete for survival. Each
individual is designated by a set of genes that define its
behaviour. Individuals that perform better (as defined by the
fitness function) have a higher chance of mating with other
individuals. When two individuals mate, they swap some of
their genes, resulting in an individual that has properties
from both of its "parents". Every now and then, a mutation
occurs where some gene randomly changes value, resulting in
a different individual. If all is well defined, after a few
generations, the population should converge on a "good-enough"
solution to the problem being tackled.

A GA implementation runs for a discrete number of time steps
called I<generations>. What happens during each generation can
vary greatly depending on the strategy being used (See 
L</"STRATEGIES"> for more info).
Typically, a variation of the following happens at
each generation:

=over 4

=item B<1. Selection>

Here the performance of all the individuals is evaluated
based on the fitness function, and each is given a specific
fitness value. The higher the value, the bigger the chance
of an individual passing its genes on in future generations
through mating (crossover).

=item B<2. Crossover>

Here, individuals selected are randomly paired up for
crossover (aka I<sexual reproduction>). This is further
controlled by the crossover rate specified and may result in
a new offspring individual that contains genes common to
both parents. New individuals are injected into the current
population.

=item B<3. Mutation>

In this step, each individual is given the chance to mutate
based on the mutation probability specified. If an individual
is to mutate, each of its genes is given the chance to randomly
switch its value to some other state.

=back

=head1 CLASS METHODS

Here are the public methods.

=over 4

=item I<$ga>-E<gt>B<new>(I<options>)

This is the constructor. It accepts options in the form of
hash-value pairs. These are:

=over 8

=item B<-population>

This defines the size of the population, i.e. how many individuals
to simultaneously exist at each generation. Defaults to 100.

=item B<-crossover>

This defines the crossover rate. Defaults to 0.95.

=item B<-mutation>

This defines the mutation rate. Defaults to 0.05.

=item I<-fitness>

This defines a fitness function. It expects a reference to a subroutine.
More details are given in L</"FITNESS FUNCTION">.

=item I<-type>

This defines the type of the genome. Currently, AI::Genetic
supports only three types:

=over

=item I<bitvector>

Individuals of this type have genes that are bits. Each gene
can be in one of two possible states, on or off.

=item I<listvector>

Each gene of a listvector individual can assume one string value from
a specified list of possible string values.

=item I<rangevector>

Each gene of a rangevector individual can assume one integer value
from a range of possible integer values. Note that only integers are
supported. The user can always transform any desired fractional values
by multiplying and dividing by an appropriate power of 10.

=back

Defaults to I<bitvector>.

=item I<-terminate>

This option allows the definition of a termination subroutine.
It expects a subroutine reference. This sub will be called at
the end of each generation with one argument: the AI::Genetic
object. Evolution terminates if the sub returns a true value.

=back

=item I<$ga>-E<gt>B<createStrategy>(I<strategy_name>, I<sub_ref>)

This method allows the creation of a custom-made strategy to be used
during evolution. It expects a unique strategy name, and a subroutine
reference as arguments. The subroutine will be called with one argument:
the AI::Genetic object. It is expected to alter the population at each
generation. See L</"STRATEGIES"> for more information.

=item I<$ga>-E<gt>B<init>(I<initArgs>)

This method initializes the population with random individuals. It B<MUST>
be called before any call to I<evolve()> or I<inject()>. As a side effect,
any already existing individuals in the population are deleted. It expects
one argument, which depends on the type of individuals:

=over

=item o

For bitvectors, the argument is simply the length of the bitvector.

    $ga->init(10);

this initializes a population where each individual has 10 genes.

=item o

For listvectors, the argument is an anonymous list of lists. The
number of sub-lists is equal to the number of genes of each individual.
Each sub-list defines the possible string values that the corresponding gene
can assume.

    $ga->init([
               [qw/red blue green/],
               [qw/big medium small/],
               [qw/very_fat fat fit thin very_thin/],
              ]);

this initializes a population where each individual has 3 genes, and each gene
can assume one of the given values.

=item o

For rangevectors, the argument is an anonymous list of lists. The
number of sub-lists is equal to the number of genes of each individual.
Each sub-list defines the minimum and maximum integer values that the
corresponding gene can assume.

    $ga->init([
               [1, 5],
               [0, 20],
               [4, 9],
              ]);

this initializes a population where each individual has 3 genes, and each gene
can assume an integer within the corresponding range.

=back

=item I<$ga>-E<gt>B<inject>(I<N>, ?I<args>?)

This method can be used to add more individuals to the population. New individuals
can be randomly generated, or be explicitly specified. The first argument specifies
the number, I<N>, of new individuals to add. This can be followed by at most I<N>
arguments, each of which is an anonymous list that specifies the genome of a
single individual to add. If the number of genomes given, I<n>, is less than I<N>, then
I<N> - I<n> random individuals are added for a total of I<N> new individuals. Random
individuals are generated using the same arguments passed to the I<init()> method.
For example:

  $ga->inject(5,
              [qw/red big thin/],
              [qw/blue small fat/],
             );

this adds 5 new individuals, 2 with the specified genetic coding, and 3 randomly
generated.

=item I<$ga>-E<gt>B<evolve>(I<strategy>, ?I<num_generations>?)

This method causes the GA to evolve the population using the specified strategy.
A strategy name has to be specified as the first argument. The second argument
is optional and specifies the number of generations to evolve. It defaults to
1. See L</"STRATEGIES"> for more information on the default strategies.

Each generation consists of the following steps:

=over

=item o

The population is sorted according to the individuals' fitnesses.

=item o

The subroutine corresponding to the named strategy is called with one argument,
the AI::Genetic object. This subroutine is expected to alter the object itself.

=item o

If a termination subroutine is given, it is executed and the return value is
checked. Evolution terminates if this sub returns a true value.

=back

=item I<$ga>-E<gt>B<getFittest>(?I<N>?)

This returns the I<N> fittest individuals. If not specified,
I<N> defaults to 1. As a side effect, it sorts the population by
fitness score. The actual AI::Genetic::Individual objects are returned.
You can use the C<genes()> and C<score()> methods to get the genes and the
scores of the individuals. Please check L<AI::Genetic::Individual> for details.

=item I<$ga>-E<gt>B<sortPopulation>

This method sorts the population according to fitness function. The results
are cached for speed.

=item I<$ga>-E<gt>B<sortIndividuals>(?[I<ListOfIndividuals>]?)

Given an anonymous list of individuals, this method sorts them according
to fitness, returning an anonymous list of the sorted individuals.

=item I<$ga>-E<gt>B<people>()

Returns an anonymous list of individuals of the current population.
B<IMPORTANT>: the actual array reference used by the AI::Genetic object
is returned, so any changes to it will be reflected in I<$ga>.

=item I<$ga>-E<gt>B<size>(?I<newSize>?)

This method is used to query and set the population size.

=item I<$ga>-E<gt>B<crossProb>(?I<newProb>?)

This method is used to query and set the crossover rate.

=item I<$ga>-E<gt>B<mutProb>(?I<newProb>?)

This method is used to query and set the mutation rate.

=item I<$ga>-E<gt>B<indType>()

This method returns the type of individual: I<bitvector>, I<listvector>,
or I<rangevector>.

=item I<$ga>-E<gt>B<generation>()

This method returns the current generation.

=back

=head1 FITNESS FUNCTION

Very quickly you will realize that properly defining the fitness function
is the most important aspect of a GA. Most of the time that a genetic
algorithm takes to run is spent in running the fitness function for each
separate individual to get its fitness. AI::Genetic tries to minimize this
time by caching the fitness result for each individual. But, B<you should
spend a lot of time optimizing your fitness function to achieve decent run
times.>

The fitness function should expect only one argument, an anonymous list of
genes, corresponding to the individual being analyzed. It is expected
to return a number which defines the fitness score of the said individual.
The higher the score, the more fit the individual, the more the chance it
has to be chosen for crossover.

=head1 STRATEGIES

AI::Genetic comes with 9 predefined strategies. These are:

=over

=item rouletteSinglePoint

This strategy implements roulette-wheel selection and single-point crossover.

=item rouletteTwoPoint

This strategy implements roulette-wheel selection and two-point crossover.

=item rouletteUniform

This strategy implements roulette-wheel selection and uniform crossover.

=item tournamentSinglePoint

This strategy implements tournament selection and single-point crossover.

=item tournamentTwoPoint

This strategy implements tournament selection and two-point crossover.

=item tournamentUniform

This strategy implements tournament selection and uniform crossover.

=item randomSinglePoint

This strategy implements random selection and single-point crossover.

=item randomTwoPoint

This strategy implements random selection and two-point crossover.

=item randomUniform

This strategy implements random selection and uniform crossover.

=back

More detail on these strategies and how to call them in your own
custom strategies can be found in L<AI::Genetic::OpSelection>,
L<AI::Genetic::OpCrossover> and L<AI::Genetic::OpMutation>.

You can use the functions defined in the above modules in your
own custom-made strategy. Consult their manpages for more info.
A custom-made strategy can be defined using the I<strategy()>
method and is called at the beginning of each generation. The only
argument to it is the AI::Genetic object itself. Note that the
population at this point is sorted accoring to each individual's
fitness score. It is expected that the strategy sub will modify
the population stored in the AI::Genetic object. Here's the
pseudo-code of events:

    for (1 .. num_generations) {
      sort population;
      call strategy_sub;
      if (termination_sub exists) {
        call termination_sub;
        last if returned true value;
      }
    }

=head1 A NOTE ON SPEED/EFFICIENCY

Genetic algorithms are inherently slow.
Perl can be pretty fast, but will never reach the speed of optimized
C code (at least my Perl coding will not). I wrote AI::Genetic mainly
for my own learning experience, but still tried to optimize it as
much as I can while trying to keep it as flexible as possible.

To do that, I resorted to some well-known tricks like passing a
reference of a long list instead of the list itself (for example,
when calling the fitness function, a reference of the gene list
is passed), and caching fitness scores (if you try to evaluate
the fitness of the same individual more than once, then the fitness
function will not be called, and the cached result is returned).

To help speed up your run times, you should pay special attention
to the design of your fitness function since this will be called once
for each unique individual in each generation. If you can shave off a
few clock cycles here and there, then it will be greatly magnified in
the total run time.

=head1 BUGS

I have tested this module quite a bit, and even used it to solve a
work-related problem successfully. But, if you think you found a bug
then please let me know, and I promise to look at it.

Also, if you have any requests, comments or suggestions, then feel
free to email me.

=head1 INSTALLATION

Either the usual:

    perl Makefile.PL
    make
    make install

or just stick it somewhere in @INC where perl can find it. It is in pure Perl.

=head1 AUTHOR & CREDITS

Written by Ala Qumsieh I<aqumsieh@cpan.org>.

Special thanks go to John D. Porter and Oliver Smith for stimulating
discussions and great suggestions. Daniel Martin and Ivan Tubert-Brohman
uncovered various bugs and for this I'm grateful.

=head1 COPYRIGHTS

(c) 2003-2005 Ala Qumsieh. All rights reserved.
This module is distributed under the same terms as Perl itself.

=cut
