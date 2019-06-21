##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2019, Intel Corporation
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

# Purpose: parse command-line options and do_command utility.

package CmdLine;

require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(process_command_line print_options_help do_command);
@EXPORT_OK = qw($debug $verbose);

use strict;

my($debug) = 0;
my($verbose) = 0;

#######################################
#
# Command line parsing routines with built-in help
# with default-settings.
#
# Example usage:
# my(@KNOBS) =
#    ( # knob,        description,   optional default
#     ["val=i", "value", '3000'],
#     ["output_dir=s", "output directory", "temp"],
#     ["fast!", "use fast mode", 0],
# );
#
# my(%OPT);
# my($command_line) = process_command_line(\%OPT, \@KNOBS);
#
# print "$command_line\n" if $OPT{verbose};
# if (!$command_line || $OPT{help} || @ARGV < 1) {
#    print "usage: $0 [options] <arg(s)>";
#    print_options_help(\@KNOBS);
#  exit 1;
# }
# my $val = $OPT{val}; # etc.
#######################################

use Getopt::Long;

sub process_command_line {
    my($OPT, $KNOBS) = @_;
    
    my $cmd_line;

    $cmd_line = "$0 ";
    
    $cmd_line .= quote_shell_cmd(@main::ARGV);

    # Standard knobs.
    push(@$KNOBS, 
	 ['help!',    'Print help', $OPT->{help}],
	 ['verbose!', 'Make all command echo output', $OPT->{verbose}],
	 ['debug!',     'Enable debugging', $OPT->{debug}]
        );
    
    my(@options, %regexes);
    # setup default values
    %Perf_env::DEFAULTS = ();
    %Perf_env::CMDLINE = ();

    my($arr);
    foreach $arr (@$KNOBS) {
      my($opt) = $arr->[0];
      $opt =~ s/\W.*//;
      if (defined($arr->[2])) {
        $Perf_env::DEFAULTS{$opt} = $arr->[2];
      }
      push(@options, $arr->[0]);
      if (defined $arr->[3]) {
        $regexes{$opt} = $arr->[3];
      }
    }
    if (!GetOptions(\%Perf_env::CMDLINE, @options)) {
	return undef;
    }

    %{$OPT} = %Perf_env::DEFAULTS;
    @{$OPT}{keys %Perf_env::CMDLINE} = values %Perf_env::CMDLINE;

    # check regexs.
    my $ok = 1;
    while (my ($opt, $value) = each %$OPT) {
      my $regex = $regexes{$opt};
      if (defined $regex && $value !~ /^($regex)$/) {
        warn "Error: value of option '$opt' does not match regular expression '$regex'.\n";
        $ok = 0;
      }
    }
    return undef if !$ok;

    $debug = $OPT->{debug};
    $verbose = $OPT->{verbose};

    return $cmd_line;
}

# When passed an array this return a string that the shell
# will parse into the same array.
# Used to print @ARGV to stdout
sub quote_shell_cmd (@) {
    my(@cmd);
    my($cmd) = '';
    
    foreach (@_) {
	if (m{[^\w/.-]} || length == 0) {
	    if (/'/) { #'
		push(@cmd, "\"$_\"");
	    } else {
		push(@cmd, "'$_'");
	    }
	} else {
	    push(@cmd, $_);
	}
    }
    return join(' ', @cmd);
}    

# When passed a string return and array that the should
# would generate if passed on the command line.
#  BUG: does not handle newlines correctly

sub unquote_shell_cmd ($) {
    my($cmd) = @_;
    my(@args);

    chomp(@args = `for arg in $cmd; do echo \$arg' '; done`);
    chop(@args);  # remove extra spaces
    @args;
}


my(%type_map) = ('s' => 'string', 'f' => 'float', 'i' => 'integer');

sub print_options_help {
    my($KNOBS) = @_;

    my($arr);
    foreach $arr (@$KNOBS) {
	my($opt) = $arr->[0];
	my($name);
	
	($name = $opt) =~ s/\W.*//;

	if ($opt =~ /\!$/) {
		print "-[no]$name";
		print "\t(default: ",
	                ($arr->[2] ? "ON" : "OFF"),
			")";
	} elsif ($opt =~ /=([sif])$/) {
		print "-$name $type_map{$1}";
		if (defined $arr->[2]) {
			print "\t(default: ", quote_shell_cmd($arr->[2]), ")";
		}
		if (defined $arr->[3]) {
			print "\t(must match ", quote_shell_cmd($arr->[3]), ")";
		}
	} elsif ($opt =~ /:([sif])$/) {
		print "-$name [$type_map{$1}]";
		if (defined $arr->[2]) {
			print "\t(default: ", quote_shell_cmd($arr->[2]), ")";
		}
		if (defined $arr->[3]) {
			print "\t(must match /", $arr->[3], "/)";
		}
	} else {
		print "-$name";
	}
	print "\n\t$arr->[1]\n";
    }
}

#######################################
# Shell Command Execution Subroutines
#######################################

sub do_command($) {
    my($command) = @_;

    if ($debug || $verbose) {
	printf("-> %s\n", $command);
    }
    
    if (! $debug) {
	my($status) = 0xffff & system($command);
	
	if ($verbose) {
	    if ($status == 0) {
		
	    } elsif ($status == 0xff00) {
		print "   Command failed: $!\n";
	    } elsif ($status > 0x80) {
		printf "    Command ran with non-zero exit status %d\n", $status >> 8;
	    } else {
		print "  Command ran with ";
		if ($status & 0x80) {
		    print "coredump from ";
		}
		printf "signal %d\n", $status & ~0x80;
	    }
	}
	return($status);
    } else {
	return(0);
    }
}


# return with a 1 so require() will not fail...
#
1;

