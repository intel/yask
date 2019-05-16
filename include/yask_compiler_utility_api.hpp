/*****************************************************************************

YASK: Yet Another Stencil Kernel
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

// This file contains a base class and macros to create
// stencils to be included in the YASK compiler binary utility.
/** @file yask_compiler_utility_api.hpp */

#pragma once

// Standard headers not provided by API.
#include <cassert>

// These functions and macros are built upon the YASK compiler APIs.
#include "yask_compiler_api.hpp"

namespace yask {

    /**
     * \defgroup ycu YASK Compiler Utility
     * Wrappers around \ref yc_solution pointers used to create solutions that will
     * be integrated into the provided `yask_compiler.exe` binary utility.
     * Not to be used by YASK stencil code that is _not_ to be used with the compiler
     * utility. In other words, if you write your own binary to generate optimized 
     * YASK stencil code, use the APIs in \ref sec_yc only.
     * @{
     */

    /// All C++ stencil solutions written for the YASK compiler utility must implement this class.
    /**
       This is a wrapper around a \ref yc_solution pointer.
       The `define()` method must be overloaded by
       the YASK DSL programmer to add stencil equations.

       Not to be used by DSL code that is not a part of the YASK compiler
       utility `yask_compiler.exe`. For DSL code not using this utility,
       call yc_factory::new_solution(), yc_solution::format(), etc. directly.
    */
    class yc_solution_base {
    private:

        /// Pointer to the YASK stencil solution.
        yc_solution_ptr _soln;

        /// Factory to create new nodes.
        yc_node_factory _node_factory;
        
    public:

        /// Commonly-used constructor.
        /**
           Creates a new yc_solution object and
           registers this object in the list for the YASK compiler.
           The define() method will be called from the YASK
           compiler for the selected solution.
        */
        yc_solution_base(const std::string& name);

        /// **[Advanced]** Constructor that uses an existing yc_solution_base to share underlying solutions.
        /** 
            This constructor allows the use of object-oriented composition
            instead of inheritance when creating classes that participate
            in solution definition.
            When using this version, the define() method will
            _not_ be called directly from the YASK compiler, but it (or any
            other method) may be called by the parent `base` object
            explicitly.
        */
        yc_solution_base(yc_solution_base& base) {
            _soln = base.get_soln();
            assert(_soln.get());
        }

        /// Destructor.
        virtual ~yc_solution_base() { }

        /// Define all functionality of this solution.
        /**
           When a stencil solution is selected by naming it via `stencil=name`
           when invoking `make` or via `-stencil name` on the YASK compiler
           command-line, this function in the named solution will be called.

           This function must be implemented by each concrete stencil solution to
           add vars and equations as needed to define the stencil.
           In general, any YASK compiler API functions may be called from this
           function.

           For DSL code not using the YASK compiler
           utility `yask_compiler.exe`, the code that would be in define() 
           could be called from `main()` or any other called function.
         */
        virtual void define() {
            yask_exception e("Error: no stencil equations are defined in solution '" +
                             get_soln()->get_name() +
                             "'. Implement the 'define()' method in the class "
                             "derived from 'yc_solution_base'");
            throw e;
        }

        /// Access the underlying solution.
        virtual yc_solution_ptr
        get_soln() {
            return _soln;
        }

        /// A simple wrapper for yc_node_factory::new_step_index().
        virtual yc_index_node_ptr
        new_step_index(const std::string& name) {
            return _node_factory.new_step_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_domain_index().
        virtual yc_index_node_ptr
        new_domain_index(const std::string& name) {
            return _node_factory.new_domain_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_misc_index().
        virtual yc_index_node_ptr
        new_misc_index(const std::string& name) {
            return _node_factory.new_misc_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_number_node().
        virtual yc_number_node_ptr
        new_number_node(yc_number_any_arg arg) {
            return _node_factory.new_number_node(arg);
        }

        /// A simple wrapper for yc_node_factory::new_first_domain_index().
        virtual yc_number_node_ptr
        first_domain_index(yc_index_node_ptr dim) {
            return _node_factory.new_first_domain_index(dim);
        }

        /// A simple wrapper for yc_node_factory::new_last_domain_index().
        virtual yc_number_node_ptr
        last_domain_index(yc_index_node_ptr dim) {
            return _node_factory.new_last_domain_index(dim);
        }
    };

    /// A base class for stencils that have a "radius" size parameter.
    /**
       For a symmetric finite-difference stencil, the "radius" is often the
       number of points in the spatial dimension(s) from the center
       point of a finite-difference approximation. However, any meaning
       may be given to this variable. For example, it could be the
       minimum or maximum radius for an asymmetical stencil.
    */
    class yc_solution_with_radius_base : public yc_solution_base {
    private:

        /// A variable that controls the size the stencil.
        int _radius;

    public:
        /// Constructor.
        yc_solution_with_radius_base(const std::string& name, int radius) :
            yc_solution_base(name) {
            set_radius(radius);
        }

        /// Define all functionality of this solution.
        /**
           See yc_solution_base::define().
        */
        virtual void
        define() override {
            yask_exception e("Error: no stencil equations are defined in solution '" +
                             get_soln()->get_name() +
                             "'. Implement the 'define()' method in the class "
                             "derived from 'yc_solution_with_radius_base'");
            throw e;
        }

        /// Set radius.
        /** @returns `true` if successful. */
        virtual bool
        set_radius(int radius) {
            _radius = radius;
            auto soln = get_soln();
            soln->set_description(soln->get_name() + " radius " + std::to_string(radius));
            return radius >= 0;  // support only non-neg. radius.
        }

        /// Get radius.
        virtual int
        get_radius() const { return _radius; }
    };

    /** @}*/

} // namespace yask.
