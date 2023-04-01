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

# Purpose: Convert old-style (YASK v2) DSL code to use only the published
# YASK compiler APIs.

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
  my $class_name;
  my $list_var;
  while (<INF>) {
    $lineno++;
    chomp;

    # Capture class name.
    if (/class\s+([_\w]+)/) {
      $class_name = $1;
      warn "  class $class_name...\n" if $OPT{verbose};
    }

    # Capture stencilList parameter name.
    if (/StencilList\s*&\s*([_\w]+)/) {
      $list_var = $1;
      warn "    StencilList $list_var...\n" if $OPT{verbose};
    }

    # Register solution.
    if (/REGISTER_STENCIL\s*[(](.*)[)]/) {
      my $cname = $1;

      $result .=
        "// Create an object of type '$cname',\n".
        "// making it available in the YASK compiler utility via the\n".
        "// '-stencil' commmand-line option or the 'stencil=' build option.\n".
        "REGISTER_SOLUTION($cname);\n";
    }

    # Include file.
    elsif (/[#]include.*Soln[.]hpp/) {
      $result .= "// YASK stencil solution(s) in this file will be integrated into the YASK compiler utility.\n".
        "#include \"yask_compiler_api.hpp\"\n".
        "using namespace std;\n".
        "using namespace yask;\n".
        "\n// Create an anonymous namespace to ensure that types are local.\n".
        "namespace {\n";
    }

    # For other code, make substitutions and append changes.
    else {

      # Ctor: remove StencilList parameter.
      if (defined $class_name && defined $list_var) {
        s/\bStencilList\s*&\s*//g;
        s/([(,]\s*)$list_var\s*,\s*/$1/g; # '(a,list,b)' -> '(a,b)'
        s/([,]\s*)$list_var\s*([)])/$2/g; # '(a,list)' -> '(a)'
        s/([(]\s*)$list_var\s*([)])/$1$2/g; # '(list)' -> '()'
      }
    
      # Node creation.
      s/\bconstNum\b/new_number_node/g;
      s/\bfirst_index\b/first_domain_index/g;
      s/\blast_index\b/last_domain_index/g;

      # Var creation.
      s/\bMAKE_GRID\b/MAKE_VAR/g;
      s/\bMAKE_SCALAR(_ARRAY)?\b/MAKE_SCALAR_VAR/g;
      s/\bMAKE_SCRATCH_(GRID|SCALAR|ARRAY)\b/MAKE_SCRATCH_VAR/g;

      # Typenames.
      s/\bStencilBase\b/yc_solution_base/g;
      s/\bStencilRadiusBase\b/yc_solution_with_radius_base/g;
      s/\bGrid\b/yc_var_proxy/g;
      s/\bGridIndex\b/yc_number_node_ptr/g;
      s/\bGridValue\b/yc_number_node_ptr/g;
      s/\bCondition\b/yc_bool_node_ptr/g;
      s/\bGridPointPtr\b/yc_var_point_node_ptr/g;
      s/\bExprPtr\b/yc_expr_node_ptr/g;
      s/\bNumExprPtr\b/yc_number_node_ptr/g;
      s/\bIndexExprPtr\b/yc_index_node_ptr/g;
      s/\bBoolExprPtr\b/yc_bool_node_ptr/g;

      # Vars.
      s/\b_radius\b/get_radius()/g;

      # Other macros.
      s/\b(EQUALS_OPER|IS_EQUIV_TO|IS_EQUIVALENT_TO)\b/EQUALS/g;
      s/\b(IF|IF_OPER)\b/IF_DOMAIN/g;
      s/\b(IF_STEP_OPER)\b/IF_STEP/g;

      # Non-convertable code.
      if (/REGISTER_STENCIL_CONTEXT_EXTENSION|StencilPart/) {
        warn "  Warning: the v2 '$&' construct on line $lineno must be manually edited.\n";
        $result .= "  ## Warning: the v2 '$&' construct cannot be automatically converted.\n".
        "  ## You must manually edit the following line(s).\n";
      }
      
      $result .= "$_\n";
    }
  }
  close INF;
  $result .= "\n} // anonymous namespace.\n";

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
    print "Converts old-style (YASK v2) DSL code to use published YASK compiler APIs.\n",
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
