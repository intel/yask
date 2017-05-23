#!/usr/bin/env python

##############################################################################
## YASK: Yet Another Stencil Kernel
## Copyright (c) 2014-2017, Intel Corporation
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

## Create the YASK stencil compiler API for Python using SWIG via the 'distutils' package.

# http://www.swig.org/Doc3.0/Python.html
# https://docs.python.org/3.6/distutils/index.html

"""
setup.py file for YASK compiler API.
"""

from distutils.core import setup, Extension

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Create an extension module for the YASK compiler API.
yask_compiler_module = Extension('_yask_compiler',
                                 sources=['yask_compiler_api_wrap.cxx',
                                          '../lib/Expr.cpp',
                                          '../lib/ExprUtils.cpp',
                                          '../lib/CppIntrin.cpp',
                                          '../lib/Print.cpp'],
                                 include_dirs=['../../../include',
                                               '../../common',
                                               '../lib'],
                                 extra_compile_args=['-std=c++11'],
)

# Invoke the setup.
setup (name = 'yask_compiler',
       version = '0.1',
       author      = "Chuck Yount",
       description = """YASK stencil compiler API""",
       ext_modules = [yask_compiler_module],
       py_modules = ["yask_compiler"],
)
