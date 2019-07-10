/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2019, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

// This file defines functions, types, and macros needed for the stencil
// kernel.

#pragma once

// Stencil-independent definitions.
#include "yask.hpp"

// Auto-generated macros from foldBuilder.  It's important that this be
// included before the definitions below to properly set the vector lengths,
// etc.
#define DEFINE_MACROS
#include YSTR2(YK_CODE_FILE)
#undef DEFINE_MACROS

// Macro to loop thru domain dims w/stencil index 'i' and domain index 'j'.
// Step index must be at index zero.
#define _DOMAIN_VAR_LOOP(i, j)                                  \
    for (int i = 1, j = 0; j < NUM_DOMAIN_DIMS; i++, j++)
#if (defined CHECK) || (defined TRACE)
#define DOMAIN_VAR_LOOP(i, j)                   \
    _DOMAIN_VAR_LOOP(i, j)
#else
#define DOMAIN_VAR_LOOP(i, j)                                   \
    _Pragma("unroll")                                           \
    _DOMAIN_VAR_LOOP(i, j)
#endif
    
// Max number of dims allowed in Indices.
// TODO: make Indices a templated class based on
// number of dims.
#ifndef MAX_DIMS
#if NUM_VAR_DIMS >= NUM_STENCIL_DIMS
#define MAX_DIMS NUM_VAR_DIMS
#else
#define MAX_DIMS NUM_STENCIL_DIMS
#endif
#endif

// First/last index macros.
// These are relative to global problem, not rank.
#define FIRST_INDEX(dim) (0)
#define LAST_INDEX(dim) (_context->get_settings().get()->_global_sizes[STENCIL_DIM_IDX_ ## dim] - 1)

// Macros for 1D<->nD transforms.
#include "yask_layout_macros.hpp"

// Define a folded vector of reals.
#include "realv.hpp"

// Base types for stencil context, etc.
#include "indices.hpp"
#include "settings.hpp"
#include "generic_var.hpp"
#include "yk_var.hpp"
#include "auto_tuner.hpp"
#include "context.hpp"
#include "stencil_calc.hpp"
