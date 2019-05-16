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

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yc_solution_api.hpp */

#pragma once

// Standard headers.
#include <cassert>
#include <map>

namespace yask {

    /**
     * \addtogroup yc
     * @{
     */

    /// A base class for defining solutions to be kept in a common registry.
    /**
       This is a wrapper around a \ref yc_solution pointer and a
       static registry used to hold all \ref yc_solution_base objects.

       This base class must be extended via inheritance.
       When using the provided YASK compiler utility,
       the `define()` method must be overloaded to add stencil equations
       and other functionality needed to implement the concrete
       solution.
    */
    class yc_solution_base {

    public:
        /// Type for a common registry shared among all yc_solution_base objects.
        typedef std::map<std::string, yc_solution_base*> soln_map;
        
    private:

        /// Pointer to the YASK stencil solution.
        yc_solution_ptr _soln;

        /// Factory to create new solutions.
        yc_factory _yc_factory;

        /// Factory to create new nodes.
        yc_node_factory _node_factory;

        /// A common registry shared among all yc_solution_base objects.
        static soln_map _soln_registry;
        
    public:

        /// Constructor.
        /**
           Creates a new yc_solution object and adds this object to the
           registry.  Throws an exception if a solution with this name
           already exists.
        */
        yc_solution_base(const std::string& name);

        /// **[Advanced]** Constructor that uses an existing yc_solution_base to share underlying solutions.
        /** 
            This constructor allows the use of object-oriented composition
            instead of inheritance when creating classes that participate
            in solution definition.
        */
        yc_solution_base(yc_solution_base& base);

        /// Destructor.
        virtual ~yc_solution_base() { }

        /// Access to the registry.
        /**
           @returns Reference to the registry.
        */
        static const soln_map& get_registry() {
            return _soln_registry;
        }

        /// Define all functionality of this solution.
        /**
           If using the provided YASK compiler utility `yask_compiler.exe`,
           when a stencil solution is selected via `stencil=name`
           when invoking `make` or via `-stencil name` on the YASK compiler
           command-line, this function in the named solution will be called.

           In this case, 
           this function must be implemented by each concrete stencil solution to
           add vars and equations as needed to define the stencil.
           In general, any YASK compiler API functions may be called from this
           function.

           For custom code using the YASK compiler library but not
           the YASK compiler utility, calling define() is possible but optional.
         */
        virtual void define();

        /// Access the underlying solution.
        inline yc_solution_ptr
        get_soln() {
            return _soln;
        }

        /// A simple wrapper for yc_node_factory::new_step_index().
        inline yc_index_node_ptr
        new_step_index(const std::string& name) {
            return _node_factory.new_step_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_domain_index().
        inline yc_index_node_ptr
        new_domain_index(const std::string& name) {
            return _node_factory.new_domain_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_misc_index().
        inline yc_index_node_ptr
        new_misc_index(const std::string& name) {
            return _node_factory.new_misc_index(name);
        }

        /// A simple wrapper for yc_node_factory::new_number_node().
        inline yc_number_node_ptr
        new_number_node(yc_number_any_arg arg) {
            return _node_factory.new_number_node(arg);
        }

        /// A simple wrapper for yc_node_factory::new_first_domain_index().
        inline yc_number_node_ptr
        first_domain_index(yc_index_node_ptr dim) {
            return _node_factory.new_first_domain_index(dim);
        }

        /// A simple wrapper for yc_node_factory::new_last_domain_index().
        inline yc_number_node_ptr
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
        define() override;

        /// Set radius and updates the solution decription.
        /** 
            The DSL programmer can overload this method to set the description
            differently and/or other side-effects.
            @returns `true` if the `radius` is valid.
        */
        virtual bool
        set_radius(int radius) {
            _radius = radius;
            auto soln = get_soln();
            soln->set_description(soln->get_name() + " radius " + std::to_string(radius));
            return radius >= 0;  // support only non-neg. radius.
        }

        /// Get radius.
        /**
           @returns current radius setting.
        */
        virtual int
        get_radius() const {
            return _radius;
        }
    };

    /** @}*/

} // namespace yask.
