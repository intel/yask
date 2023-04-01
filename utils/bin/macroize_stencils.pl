#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2023, Intel Corporation
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

# Purpose: Convert stencil code to use convenience macros.

use strict;
use File::Basename;
use File::Path;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

use File::Which;
use Text::ParseWords;
use FileHandle;
use CmdLine;

$| = 1;                         # autoflush.

# Globals.
my %OPT;                        # cmd-line options.

sub convert($) {
  my $fname = shift;

  open INF, "<$fname" or die("error: cannot open '$fname'\n");
  warn "Converting '$fname'...\n";

  # Read old file and save conversion in a string.
  my $result;
  my $lineno = 0;
  while (<INF>) {
      $lineno++;
      chomp;

      # Index vars.
      if (/(yc_index_node_ptr|auto)\s+(\w+)\s*=\s*new_(\w+)_index\s*\(\s*"(\w+)"\s*\)/) {
          my ($vname, $itype, $iname) = ($2, $3, $4);
          my ($pre, $post) = ($`, $');
          if ($vname eq $iname) {
              if ($itype eq "step" ||
                  $itype eq "domain" ||
                  $itype eq "misc") {
                  my $mac = "MAKE_".uc($itype)."_INDEX";
                  $_ = "${pre}$mac($iname)$post";
              } else {
                  warn "in line $lineno, '$_': unknown index type; cannot use macro\n";
              }
          } else {
              warn "in line $lineno, '$_': var name != index name; cannot use macro\n";
          }
      }
      elsif (/new_(\w+)_index/) {
              warn "in line $lineno, '$_': cannot convert\n";
      }

      # Vars.
      elsif (/yc_var_proxy\s+(\w+)\s*=\s*yc_var_proxy\s*\(\s*"(\w+)"\s*,\s*get_soln\s*\(\s*\)(\s*,\s*{.*}\s*)?(,\s*(true|false)\s*)?\)/ ||
          /yc_var_proxy\s+(\w+)\s*\(\s*"(\w+)"\s*,\s*get_soln\s*\(\s*\)(\s*,\s*{.*}\s*)?(,\s*(true|false)\s*)?\)/) {
          my ($vname, $yvname, $dims, $is_scratch) = ($1, $2, $3, $4);
          my ($pre, $post) = ($`, $');
          if ($vname eq $yvname) {
              $dims =~ s/\s+//g;
              $dims =~ s/^,\{//;
              $dims =~ s/\}$//;
              $dims =~ s/,/, /g;
              my $do_scratch = (defined $is_scratch && $is_scratch =~ /true/);
              if (!$do_scratch && $dims eq "") {
                  $_ = "${pre}MAKE_SCALAR_VAR($vname)$post";
              } else {
                  my $mac = $do_scratch ? "MAKE_SCRATCH_VAR" : "MAKE_VAR";
                  $_ = "${pre}$mac($vname, $dims)$post";
              }
          } else {
              warn "in line $lineno, '$_': var name != YASK var name; cannot use macro\n";
          }
      }
      elsif (/yc_var_proxy.*get_soln/) {
              warn "in line $lineno, '$_': cannot convert\n";
      }

      # Registration.
      elsif (/static\s+(\w+)\s+(\w+)_instance/) {
          my ($sname, $iname) = ($1, $2);
          if ($sname eq $iname) {
              $_ = "$`REGISTER_SOLUTION($sname)$'"
          } else {
              warn "in line $lineno, '$_': class name != instance name; cannot use macro\n";
          }
 
      }
     
      $result .= "$_\n";
  }
  close INF;

  if ($OPT{in_place}) {

    # Backup original.
    my $fbak = $fname."~";
    rename "$fname", $fbak or die("error: cannot rename original file to '$fbak'\n");
    warn "  Original code saved in '$fbak'.\n";

    # Write new code to original filename.
    open OUTF, ">$fname" or die("error: cannot write back to '$fname'\n");
    print OUTF $result;
    close OUTF;
    warn "  Converted code written back to '$fname'.\n".
      "  Complete conversion is not guaranteed; please review and test changes.\n";
  }

  # not in-place; print to stdout.
  else {
    print $result;
  }
}

# Parse arguments and emit code.
sub main() {

  my(@KNOBS) =
    (
     # knob,        description,   optional default
     [ "in_place!", "Modify the file(s) in-place. If false, write to stdout.", 1 ],
    );
  my($command_line) = process_command_line(\%OPT, \@KNOBS);
  print "$command_line\n" if $OPT{verbose};

  my $script = basename($0);
  if (!$command_line || $OPT{help} || @ARGV < 1) {
    print "Add convenience macros to YASK DSL code.\n",
      "Usage: $script [options] <file-name(s)>\n",
      "Options:\n";
    print_options_help(\@KNOBS);
    exit 1;
  }

  for my $fname (@ARGV) {
    convert($fname);
  }
}

main();
