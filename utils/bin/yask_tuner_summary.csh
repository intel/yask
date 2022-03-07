#!/bin/csh -f

##############################################################################
## YASK: Yet Another Stencil Kit
## Copyright (c) 2014-2022, Intel Corporation
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

# Purpose: find best result from each GA search csv file.

if ( "-$1" == "-" ) then
    if ( `echo logs/yask_*/*.csv | wc -l` > 0 ) then
        $0 logs/yask_*/*.csv
    else
        echo "usage: $0 [csv-file(s) from yask_tuner.pl]"
    endif
    exit
endif

echo "Summary of yask_tuner results:"
head -n1 $1
foreach f ($*)
    echo '=========='
    wc -l $f
    grep ',TRUE' $f | tail -n1
end

