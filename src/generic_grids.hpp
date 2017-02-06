/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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
#include "utils.hpp"

namespace yask {

#include "layouts.hpp"

    // A base class for a generic grid of elements of arithmetic type T.
    // This class provides linear-access support, i.e., no layout.
    template <typename T> class GenericGridBase {
    protected:
        T* _elems = 0;
        bool _do_free = false;
        const static size_t _def_alignment = CACHELINE_BYTES;

    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridBase() { }

        // Dealloc memory only if allocated via default_alloc().
        virtual ~GenericGridBase() {
            if (_elems && _do_free)
                free(_elems);
        }

        // Perform default allocation. Will be free'd upon destruction.
        // For other options,
        // programmer should call get_num_elems() or get_num_bytes() and
        // then provide allocated memory via set_storage().
        virtual void default_alloc() {
            size_t sz = get_num_bytes();
            int ret = posix_memalign((void **)&_elems, _def_alignment, sz);
            if (ret) {
                std::cerr << "error: cannot allocate " << sz << " bytes." << std::endl;
                exit_yask(1);
            }
            _do_free = true;
        }
        
        // Get number of elements.
        virtual idx_t get_num_elems() const =0;

        // Get size in bytes.
        inline size_t get_num_bytes() const {
            return sizeof(T) * get_num_elems();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
            os << "'" << name << "' data is at " << _elems << ", containing " <<
                printWithPow10Multiplier(get_num_elems()) << " element(s) of " <<
                sizeof(T) << " byte(s) each = " <<
                printWithPow2Multiplier(get_num_bytes()) << " bytes.\n";
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

        // Direct access to data.
        T* get_storage() {
            return _elems;
        }
        const T* get_storage() const {
            return _elems;
        }

        // Set pointer to storage.
        // Free old storage if it was allocated in ctor.
        // 'buf' should provide get_num_bytes() bytes at offset bytes.
        void set_storage(void* buf, size_t offset) {
            if (_elems && _do_free) {
                free(_elems);
                _elems = 0;
            }
            _do_free = false;
            char* p = static_cast<char*>(buf) + offset;
            _elems = (T*)(p);
        }
    };

    // A generic 0D grid (scalar) of elements of type T.
    // No layout function needed, because there is only 1 element.
    template <typename T> class GenericGrid0d :
        public GenericGridBase<T> {
    
    public:

        // Construct an unallocated scalar.
        GenericGrid0d() {}

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
            os << "scalar ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return 1;
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
                    os << "** mismatch: " <<
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
        LayoutFn _layout;
    
    public:

        // Construct an unallocated array of length 1.
        GenericGrid1d() { }

        // Construct an array of length d1.
        GenericGrid1d(idx_t d1) :
            _layout(d1) { }

        // Get/set size.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline void set_d1(idx_t d1) { _layout.set_d1(d1); }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
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
        LayoutFn _layout;
    
    public:

        // Construct an unallocated grid of dimensions 1*1.
        GenericGrid2d() { }

        // Construct a grid of dimensions d1*d2.
        GenericGrid2d(idx_t d1, idx_t d2) :
            _layout(d1, d2) { }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline void set_d1(idx_t d1) { _layout.set_d1(d1); }
        inline void set_d2(idx_t d2) { _layout.set_d2(d2); }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
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

        // Construct an unallocated grid of dimensions 1*1*1.
        GenericGrid3d() { }

        // Construct a grid of dimensions d1*d2*d3.
        GenericGrid3d(idx_t d1, idx_t d2, idx_t d3) :
            _layout(d1, d2, d3) { }
    
        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline void set_d1(idx_t d1) { _layout.set_d1(d1); }
        inline void set_d2(idx_t d2) { _layout.set_d2(d2); }
        inline void set_d3(idx_t d3) { _layout.set_d3(d3); }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
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
        LayoutFn _layout;
    
    public:

        // Construct an unallocated grid of dimensions 1*1*1*1.
        GenericGrid4d() { }

        // Construct a grid of dimensions d1 * d2 * d3 * d4.
        GenericGrid4d(idx_t d1, idx_t d2, idx_t d3, idx_t d4) :
            _layout(d1, d2, d3, d4) { }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline idx_t get_d4() const { return _layout.get_d4(); }
        inline void set_d1(idx_t d1) { _layout.set_d1(d1); }
        inline void set_d2(idx_t d2) { _layout.set_d2(d2); }
        inline void set_d3(idx_t d3) { _layout.set_d3(d3); }
        inline void set_d4(idx_t d4) { _layout.set_d4(d4); }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
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

    // A generic 5D grid of elements of type T.
    // The LayoutFn class must provide a 1:1 transform between
    // 5D and 1D indices.
    template <typename T, typename LayoutFn> class GenericGrid5d :
        public GenericGridBase<T> {
    protected:
        LayoutFn _layout;
    
    public:

        // Construct an unallocated grid of dimensions 1*1*1*1*1.
        GenericGrid5d() { }

        // Construct a grid of dimensions d1 * d2 * d3 * d4 * d5.
        GenericGrid5d(idx_t d1, idx_t d2, idx_t d3, idx_t d4, idx_t d5) :
            _layout(d1, d2, d3, d4, d5) { }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline idx_t get_d4() const { return _layout.get_d4(); }
        inline idx_t get_d5() const { return _layout.get_d5(); }
        inline void set_d1(idx_t d1) { _layout.set_d1(d1); }
        inline void set_d2(idx_t d2) { _layout.set_d2(d2); }
        inline void set_d3(idx_t d3) { _layout.set_d3(d3); }
        inline void set_d4(idx_t d4) { _layout.set_d4(d4); }
        inline void set_d5(idx_t d5) { _layout.set_d5(d5); }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
        }

        // Print some info.
        virtual void print_info(const std::string& name, std::ostream& os) {
            os << "5D (" << get_d1() << " * " << get_d2() << " * " <<
                get_d3() << " * " << get_d4() <<  " * " << get_d5() << ") ";
            GenericGridBase<T>::print_info(name, os);
        }

        // Get 1D index.
        inline idx_t get_index(idx_t i, idx_t j, idx_t k, idx_t l, idx_t m,
                               bool check=true) const {
            if (check) {
                assert(i >= 0);
                assert(i < get_d1());
                assert(j >= 0);
                assert(j < get_d2());
                assert(k >= 0);
                assert(k < get_d3());
                assert(l >= 0);
                assert(l < get_d4());
                assert(m >= 0);
                assert(m < get_d5());
            }
            idx_t ai = _layout.layout(i, j, k, l, m);
            if (check)
                assert(ai < _layout.get_size());
            return ai;
        }

        // Access element given 5D indices.
        inline const T& operator()(idx_t i, idx_t j, idx_t k, idx_t l, idx_t m,
                                   bool check=true) const {
            return this->_elems[get_index(i, j, k, l, m, check)];
        }

        // Non-const version.
        inline T& operator()(idx_t i, idx_t j, idx_t k, idx_t l, idx_t m,
                             bool check=true) {
            return this->_elems[get_index(i, j, k, l, m, check)];
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon.
        virtual idx_t compare(const GenericGridBase<T>* ref,
                              T epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            auto ref1 = dynamic_cast<const GenericGrid5d*>(ref);
            if (!ref1) {
                os << "** type mismatch against GenericGrid5d." << std::endl;
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
                                for (idx_t i5 = 0; i5 < get_d5(); i5++) {
                                    T te = (*this)(i1, i2, i3, i4, i5);
                                    T re = (*ref1)(i1, i2, i3, i4, i5);
                                    if (!within_tolerance(te, re, epsilon)) {
                                        p++;
                                        if (p < maxPrint)
                                            os << "** mismatch at (" << i1 << ", " << i2 <<
                                                ", " << i3 << ", " << i4 <<  ", " << i5 << "): " <<
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
            }

        done:
            return errs;
        }
    };

}

#endif
