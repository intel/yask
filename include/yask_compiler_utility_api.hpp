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
     * \addtogroup yc
     * @{
     */

    /// The class all C++ stencil solutions written for the YASK compiler
    /// utility `yask_compiler.exe` must implement.
    /**
       Mostly, this is a wrapper around a \ref yc_solution pointer.
       The `define()` method must be overloaded by
       the YASK DSL programmer to add stencil equations.

       Not to be used by DSL code that is not a part of the YASK compiler
       utility `yask_compiler.exe`. For DSL code not using this utility,
       call yc_factory::new_solution directly.
    */
    class yc_solution_base {
    protected:

        // Pointer to the YASK stencil solution.
        yc_solution_ptr _soln;

        // Factory to create new nodes.
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

        /// **[Advanced]** Constructor that uses an existing
        /// yc_solution_base to share underlying solutions.
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
            _soln = base.get_solution();
        }

        /// Destructor.
        virtual ~yc_solution_base() { }

        /// Define all functionality of this solution.
        /**
           When a stencil solution is selected by naming it via `stencil=name`
           when invoking `make` or via `-stencil name` on the YASK compiler
           command-line, this function in the named solution will be called.

           This function must be implemented by each concrete stencil solution to
           add grids and grid-value equations as needed to define the stencil.
           In general, any YASK compiler API functions may be called from this
           function.

           For DSL code not using the YASK compiler
           utility `yask_compiler.exe`, the code that would be in define() 
           could be called from `main()` or any other called function.
         */
        virtual void define() {
            std::cout << "Warning: no stencil equations are defined in solution '" <<
                _soln->get_name() << "'. Implement the 'define()' method in the class "
                "derived from 'yc_solution_base'.\n";
        }

        /// Access the underlying solution.
        virtual yc_solution_ptr get_solution() {
            return _soln;
        }

        /// Create boundary index expression, e.g., 'first_index(x)'.
        virtual yc_number_node_ptr first_index(yc_index_node_ptr dim) {
            return _node_factory.new_first_domain_index(dim);
        }

        /// Create boundary index expression, e.g., 'last_index(x)'.
        virtual yc_number_node_ptr last_index(yc_index_node_ptr dim) {
            return _node_factory.new_last_domain_index(dim);
        }

        /// This solution does _not_ use the "radius" sizing feature.
        virtual bool uses_radius() const { return false; }

        /// Dummy function for setting radius.
        virtual bool set_radius(int radius) { return false; }

        /// Dummy function for accessing radius.
        virtual int get_radius() const { return 0; }
    };

    /// A base class for stencils that have a 'radius'.
    class yc_solution_with_radius_base : public yc_solution_base {
    protected:

        // A variable that controls the size the stencil, i.e., the number of
        // points that are read to calculate a new value.  In many cases,
        // this is the number of points in the spatial dimension(s) from the
        // center point of a finite-difference approximation, but it does
        // not have to be. For example, it could be the minimum or maximum
        // radius for an asymmetical stencil.
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
        virtual void define() override {
            std::cout << "Warning: no stencil equations are defined in solution '" <<
                _soln->get_name() << "'. Implement the 'define()' method in the class "
                "derived from 'yc_solution_with_radius_base'.\n";
        }

        /// This object does use radius.
        virtual bool uses_radius() const override { return true; }

        /// Set radius.
        /** @returns `true` if successful. */
        virtual bool set_radius(int radius) override {
            _radius = radius;
            _soln->set_description(_soln->get_name() + " radius " + std::to_string(radius));
            return radius >= 0;  // support only non-neg. radius.
        }

        /// Get radius.
        virtual int get_radius() const override { return _radius; }
    };

    /** @}*/

} // namespace yask.

/// Convenience macro for declaring a static object of a type derived from
/// \ref yask::yc_solution_base and registering it in the list used by the
/// provided YASK compiler utility.
/** The derived class must implement a default constructor. */
#define YASK_REGISTER_SOLUTION(class_name) \
    static class_name registered_ ## class_name()
