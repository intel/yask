#!/usr/bin/env perl

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2024, Intel Corporation
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

# Purpose: Convert YASK log file(s) to csv format.

use strict;
use File::Basename;
use File::Path;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";
use FileHandle;
use YaskUtils;
use POSIX;

if (!@ARGV) {
  die "Usage: $0 <log file(s) from yask.sh>\n".
    "CSV output is written to STDOUT.\n".
    "Follow command with '| sort -t, -g' to sort output by performance.\n";
}

my $outFH = new FileHandle;
$outFH = *STDOUT;

# Header.
YaskUtils::printCsvHeader($outFH);
print $outFH ",date & time,log file\n";

# Values from files.
for my $arg (@ARGV) {
  for my $fn (glob $arg) {
    my %results;
    YaskUtils::getResultsFromFile(\%results, $fn);

    my $datestr = "";
    if ($fn =~ /(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})/) {
      $datestr = "$2/$3/$1 $4:$5:$6"; # format for Excel.
    }

    YaskUtils::printCsvValues(\%results, $outFH);
    print $outFH ",\"$datestr\",\"$fn\"\n";
  }
}
