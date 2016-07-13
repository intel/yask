/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

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

// Generic grids:
// T: type stored in grid.
// LayoutFn: class that transforms N dimensions to 1.

#ifndef GENERIC_GRIDS
#define GENERIC_GRIDS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdexcept>

#include <string>
#include <iostream>

namespace yask {

#include "layouts.hpp"

    // Some utility functions.
    extern double getTimeInSecs();
    extern idx_t roundUp(idx_t dim, idx_t mult, std::string name);
    extern std::string printWithPow2Multiplier(double num);
    extern std::string printWithPow10Multiplier(double num);

    // A base class for a generic grid of elements of type T.
    // This class provides linear-access support, i.e., no layout.
    template <typename T> class GenericGridBase {
    protected:
        T* _elems;
        const idx_t _num_elems;
        const static size_t _def_alignment = 64;

    public:
        GenericGridBase(idx_t num_elems, size_t alignment=_def_alignment) :
            _num_elems(num_elems)
        {
            size_t sz = sizeof(T) * num_elems;
            int ret = posix_memalign((void **)&_elems, alignment, sz);
            if (ret) {

                // TODO: provide option to throw an exception.
                std::cerr << "error: cannot allocate " << sz << " bytes." << std::endl;
                exit(1);
            }
        }

        // Dealloc memory.
        virtual ~GenericGridBase() {
            free(_elems);
        }

        // Get number of elements with padding.
        inline idx_t get_num_elems() const {
            return _num_elems;
        }

        // Get size in bytes.
        inline idx_t get_num_bytes() const {
            return sizeof(T) * _num_elems;
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "grid '" << name << "' allocation at " << _elems << " for " <<
                printWithPow2Multiplier(get_num_elems()) << " element(s) of " <<
                sizeof(T) << " byte(s) each (bytes): " <<
                printWithPow2Multiplier(get_num_bytes()) << std::endl;
        }

        // Initialize memory to a given value.
        virtual void set_same(T val) {

#pragma omp parallel for
            for (idx_t ai = 0; ai < get_num_elems(); ai++)
                _elems[ai] = val;
        }

        // Initialize memory: first element to value,
        // second to 2*value, etc.; wrap around
        // occasionally to avoid large numbers.
        virtual void set_diff(T val) {
            const idx_t wrap = 71; // prime number is good to use.

            //cout << "set_diff(" << val << "): ";
        
#pragma omp parallel for
            for (idx_t ai = 0; ai < get_num_elems(); ai++)
                _elems[ai] = val * T(ai % wrap + 1);

            //cout << "_elems[0] = " << _elems[0] << std::endl;
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t count_diffs(const GenericGridBase<T>& ref, T epsilon) const {
            idx_t errs = 0;

            // Count abs diffs > epsilon.
#pragma omp parallel for reduction(+:errs)
            for (idx_t ai = 0; ai < get_num_elems(); ai++) {
                if (!within_tolerance(_elems[ai], ref._elems[ai], epsilon))
                    errs++;
            }

            return errs;
        }

        // Compare for equality within epsilon.
        // Return number of miscompares.
        virtual idx_t compare(const GenericGridBase* ref, T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const =0;

        // Direct access to data (dangerous!).
        T* getRawData() {
            return _elems;
        }
        const T* getRawData() const {
            return _elems;
        }
    
    };

    // A generic 0D grid (scalar) of elements of type T.
    // No layout function needed, because there is only 1 element.
    template <typename T> class GenericGrid0d :
        public GenericGridBase<T> {
    
    public:

        // Construct.
        GenericGrid0d(size_t alignment=GenericGridBase<T>::_def_alignment) :
            GenericGridBase<T>(1, alignment) {}

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "Scalar ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Access element.
        inline const T& operator()(bool check=true) const {
            return this->_elems[0];
        }

        // Non-const version.
        inline T& operator()(bool check=true) {
            return this->_elems[0];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref, T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid0d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid0d." << std::endl;
                return 1;
            }

            // Quick check for errors.
            idx_t errs = GenericGridBase<T>::count_diffs(*ref, epsilon);

            // Run detailed comparison if any errors found.
            if (errs > 0 && maxPrint) {
                T te = (*this)();
                T re = (*ref1)();
                if (!within_tolerance(te, re, epsilon)) {
                    os << "** scalar mismatch: " <<
                        te << " != " << re << std::endl;
                }
            }

            return errs;
        }
    
    };

    // A generic 1D grid (array) of elements of type T.
    // The LayoutFn class must provide a 1:1 transform between
    // 1D and 1D indices (usually trivial).
    template <typename T, typename LayoutFn> class GenericGrid1d :
        public GenericGridBase<T> {
    protected:
        const LayoutFn _layout;
    
    public:

        // Construct an array of length d1.
        GenericGrid1d(idx_t d1,
                      size_t alignment=GenericGridBase<T>::_def_alignment) :
            GenericGridBase<T>(d1, alignment),
            _layout(d1) { }

        // Get original parameters.
        inline idx_t get_d1() const { return _layout.get_d1(); }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "1D (" << get_d1() << ") ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get 1D index.
        inline idx_t get_index(idx_t i, bool check=true) const {
            if (check) {
                assert(i >= 0);
                assert(i < get_d1());
            }
            idx_t ai = _layout.layout(i);
            if (check)
                assert(ai < _layout.get_size());
            return ai;
        }

        // Access element.
        inline const T& operator()(idx_t i, bool check=true) const {
            return this->_elems[get_index(i, check)];
        }

        // Non-const version.
        inline T& operator()(idx_t i, bool check=true) {
            return this->_elems[get_index(i, check)];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref, T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid1d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid1d." << std::endl;
                return 1;
            }

            // Quick check for errors.
            idx_t errs = GenericGridBase<T>::count_diffs(*ref, epsilon);

            // Run detailed comparison if any errors found.
            if (errs > 0 && maxPrint) {
                int p = 0;
                for (idx_t i1 = 0; i1 < get_d1(); i1++) {
                    T te = (*this)(i1);
                    T re = (*ref1)(i1);
                    if (!within_tolerance(te, re, epsilon)) {
                        p++;
                        if (p < maxPrint)
                            os << "** mismatch at (" << i1 << "): " <<
                                te << " != " << re << std::endl;
                        else if (p == maxPrint)
                            os << "** Additional errors not printed." << std::endl;
                        else
                            goto done;
                    }
                }
            }

        done:
            return errs;
        }
    
    };

    // A generic 2D grid of elements of type T.
    // The LayoutFn class must provide a 1:1 transform between
    // 2D and 1D indices.
    template <typename T, typename LayoutFn> class GenericGrid2d :
        public GenericGridBase<T> {
    protected:
        const LayoutFn _layout;
    
    public:


        // Construct a grid of dimensions d1 x d2.
        GenericGrid2d(idx_t d1, idx_t d2,
                      size_t alignment=GenericGridBase<T>::_def_alignment) :
            GenericGridBase<T>(d1 * d2, alignment),
            _layout(d1, d2) { }

        // Get original parameters.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "2D (" << get_d1() << " * " << get_d2() << ") ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get 1D index.
        inline idx_t get_index(idx_t i, idx_t j, bool check=true) const {
            if (check) {
                assert(i >= 0);
                assert(i < get_d1());
                assert(j >= 0);
                assert(j < get_d2());
            }
            idx_t ai = _layout.layout(i, j);
            if (check)
                assert(ai < _layout.get_size());
            return ai;
        }

        // Access element given 2D indices.
        inline const T& operator()(idx_t i, idx_t j, bool check=true) const {
            return this->_elems[get_index(i, j, check)];
        }

        // Non-const version.
        inline T& operator()(idx_t i, idx_t j, bool check=true) {
            return this->_elems[get_index(i, j, check)];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref,
                              T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid2d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid2d." << std::endl;
                return 1;
            }

            // Quick check for errors.
            idx_t errs = GenericGridBase<T>::count_diffs(*ref, epsilon);

            // Run detailed comparison if any errors found.
            if (errs > 0 && maxPrint) {
                int p = 0;
                for (idx_t i1 = 0; i1 < get_d1(); i1++) {
                    for (idx_t i2 = 0; i2 < get_d2(); i2++) {
                        T te = (*this)(i1, i2);
                        T re = (*ref1)(i1, i2);
                        if (!within_tolerance(te, re, epsilon)) {
                            p++;
                            if (p < maxPrint)
                                os << "** mismatch at (" << i1 << ", " << i2 << "): " <<
                                    te << " != " << re << std::endl;
                            else if (p == maxPrint)
                                os << "** Additional errors not printed." << std::endl;
                            else
                                goto done;
                        }
                    }
                }
            }

        done:
            return errs;
        }
    
    };

    // A generic 3D grid of elements of type T.
    // The LayoutFn class must provide a 1:1 transform between
    // 3D and 1D indices.
    template <typename T, typename LayoutFn> class GenericGrid3d :
        public GenericGridBase<T> {
    protected:
        LayoutFn _layout;
    
    public:

        // Construct a grid of dimensions d1*d2*d3.
        GenericGrid3d(idx_t d1, idx_t d2, idx_t d3,
                      size_t alignment=GenericGridBase<T>::_def_alignment) :
            GenericGridBase<T>(d1 * d2 * d3, alignment),
            _layout(d1, d2, d3) { }
    
        // Get original parameters.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "3D (" << get_d1() << " * " << get_d2() << " * " << get_d3() << ") ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get 1D index.
        inline idx_t get_index(idx_t i, idx_t j, idx_t k, bool check=true) const {
            if (check) {
                assert(i >= 0);
                assert(i < get_d1());
                assert(j >= 0);
                assert(j < get_d2());
                assert(k >= 0);
                assert(k < get_d3());
            }
            idx_t ai = _layout.layout(i, j, k);
            if (check)
                assert(ai < _layout.get_size());
            return ai;
        }

        // Access element given 3D indices.
        inline const T& operator()(idx_t i, idx_t j, idx_t k, bool check=true) const {
            return this->_elems[get_index(i, j, k, check)];
        }

        // Non-const version.
        inline T& operator()(idx_t i, idx_t j, idx_t k, bool check=true) {
            return this->_elems[get_index(i, j, k, check)];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref,
                              T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid3d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid3d." << std::endl;
                return 1;
            }

            // Quick check for errors.
            idx_t errs = GenericGridBase<T>::count_diffs(*ref, epsilon);

            // Run detailed comparison if any errors found.
            if (errs > 0 && maxPrint) {
                int p = 0;
                for (idx_t i1 = 0; i1 < get_d1(); i1++) {
                    for (idx_t i2 = 0; i2 < get_d2(); i2++) {
                        for (idx_t i3 = 0; i3 < get_d3(); i3++) {
                            T te = (*this)(i1, i2, i3);
                            T re = (*ref1)(i1, i2, i3);
                            if (!within_tolerance(te, re, epsilon)) {
                                p++;
                                if (p < maxPrint)
                                    os << "** mismatch at (" << i1 << ", " << i2 <<
                                        ", " << i3 << "): " <<
                                        te << " != " << re << std::endl;
                                else if (p == maxPrint)
                                    os << "** Additional errors not printed." << std::endl;
                                else
                                    goto done;
                            }
                        }
                    }
                }
            }

        done:
            return errs;
        }
    
    };

    // A generic 4D grid of elements of type T.
    // The LayoutFn class must provide a 1:1 transform between
    // 4D and 1D indices.
    template <typename T, typename LayoutFn> class GenericGrid4d :
        public GenericGridBase<T> {
    protected:
        const LayoutFn _layout;
    
    public:

        // Construct a grid of dimensions d1 * d2 * d3 * d4.
        GenericGrid4d(idx_t d1, idx_t d2, idx_t d3, idx_t d4,
                      size_t alignment=GenericGridBase<T>::_def_alignment) :
            GenericGridBase<T>(d1 * d2 * d3 * d4, alignment),
            _layout(d1, d2, d3, d4) { }

        // Get original parameters.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline idx_t get_d4() const { return _layout.get_d4(); }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
            os << "4D (" << get_d1() << " * " << get_d2() << " * " << get_d3() << " * " << get_d4() << ") ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get 1D index.
        inline idx_t get_index(idx_t i, idx_t j, idx_t k, idx_t l, bool check=true) const {
            if (check) {
                assert(i >= 0);
                assert(i < get_d1());
                assert(j >= 0);
                assert(j < get_d2());
                assert(k >= 0);
                assert(k < get_d3());
                assert(l >= 0);
                assert(l < get_d4());
            }
            idx_t ai = _layout.layout(i, j, k, l);
            if (check)
                assert(ai < _layout.get_size());
            return ai;
        }

        // Access element given 4D indices.
        inline const T& operator()(idx_t i, idx_t j, idx_t k, idx_t l, bool check=true) const {
            return this->_elems[get_index(i, j, k, l, check)];
        }

        // Non-const version.
        inline T& operator()(idx_t i, idx_t j, idx_t k, idx_t l, bool check=true) {
            return this->_elems[get_index(i, j, k, l, check)];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref,
                              T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid4d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid4d." << std::endl;
                return 1;
            }

            // Quick check for errors.
            idx_t errs = GenericGridBase<T>::count_diffs(*ref, epsilon);

            // Run detailed comparison if any errors found.
            if (errs > 0 && maxPrint) {
                int p = 0;
                for (idx_t i1 = 0; i1 < get_d1(); i1++) {
                    for (idx_t i2 = 0; i2 < get_d2(); i2++) {
                        for (idx_t i3 = 0; i3 < get_d3(); i3++) {
                            for (idx_t i4 = 0; i4 < get_d4(); i4++) {
                                T te = (*this)(i1, i2, i3, i4);
                                T re = (*ref1)(i1, i2, i3, i4);
                                if (!within_tolerance(te, re, epsilon)) {
                                    p++;
                                    if (p < maxPrint)
                                        os << "** mismatch at (" << i1 << ", " << i2 <<
                                            ", " << i3 << ", " << i4 << "): " <<
                                            te << " != " << re << std::endl;
                                    else if (p == maxPrint)
                                        os << "** Additional errors not printed." << std::endl;
                                    else
                                        goto done;
                                }
                            }
                        }
                    }
                }
            }

        done:
            return errs;
        }
    
    };

}

#endif
