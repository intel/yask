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
// Mapfn: class that maps N dimensions to 1.

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

#include "maps.hpp"

// A base class for a generic grid of elements of type T.
template <typename T> class GenericGridBase {
protected:
    T* _elems;
    const size_t _num_elems;
    const static size_t _def_alignment = 64;

public:
    GenericGridBase(size_t num_elems, size_t alignment=_def_alignment) :
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
    inline size_t get_num_elems() const {
        return _num_elems;
    }

    // Get size in bytes.
    inline size_t get_num_bytes() const {
        return sizeof(T) * _num_elems;
    }

    // Print some info.
    virtual void print_info(const std::string& name, std::ostream& os = std::cout) {
        const double oneK = 1024.;
        const double oneM = oneK * oneK;
        const double oneG = oneK * oneM;
        double nb = double(get_num_bytes());
        os << "Grid '" << name << "' allocated at " << _elems <<
            " with " << get_num_elems() << " elements in ";
        if (nb > oneG)
            os << (nb / oneG) << "G";
        else if (nb > oneM)
            os << (nb / oneM) << "M";
        else if (nb > oneK)
            os << (nb / oneK) << "K";
        else
            os << nb;
        os << " byte(s)" << std::endl;
    }

    // Initialize memory to a given value.
    virtual void set_same(T val) {
#pragma omp parallel for
        for (size_t ai = 0; ai < get_num_elems(); ai++)
            _elems[ai] = val;
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    virtual size_t count_diffs(const GenericGridBase<T>& ref, T epsilon) const {
        size_t errs = 0;

        // Count abs diffs > epsilon.
#pragma omp parallel for reduction(+:errs)
        for (size_t ai = 0; ai < get_num_elems(); ai++) {
            T err = _elems[ai] - ref._elems[ai];
            if (err > epsilon || err < -epsilon)
                errs++;
        }

        return errs;
    }

    // Direct access to data (dangerous!).
    T* getRawData() {
        return _elems;
    }
    const T* getRawData() const {
        return _elems;
    }
    
};

// A generic 1D grid (array) of elements of type T.
// Supports padding.
// The Mapfn class must provide a 1:1 mapping between
// 1D and 1D indices.
// This is just a degenerate case for completeness.
template <typename T, typename Mapfn> class GenericGrid1d :
    public GenericGridBase<T> {
protected:
    const Mapfn _mapfn;
    const int _d1;
    const int _p1;
    
public:

    // Construct a grid of dimensions
    // d1+2*p1,
    // i.e., with padding.
    GenericGrid1d(int d1, int p1,
           size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGridBase<T>(d1 + 2*p1, alignment),
        _mapfn(d1 + 2*p1),
        _d1(d1), _p1(p1) { }

    // Construct a grid of dimensions
    // d1, i.e., no padding.
    GenericGrid1d(int d1,
           size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGrid1d(d1, 0, alignment) { } 

    // Get original parameters.
    inline int get_d1() const { return _d1; }
    inline int get_p1() const { return _p1; }

    // Get 1D index.
    // Thus, -p1 <= i < d1+p1.
    inline size_t get_index(int i, bool check=true) const {

        // Adjust for padding.
        int ip = i + _p1;
        if (check) {
            assert(ip >= 0);
            assert(ip < _mapfn.get_d1());
        }
        size_t ai = _mapfn.map(ip);
        if (check)
            assert(ai < _mapfn.get_size());
        return ai;
    }

    // Access element.
    inline const T& operator()(int i, bool check=true) const {
        return this->_elems[get_index(i, check)];
    }

    // Non-const version.
    inline T& operator()(int i, bool check=true) {
        return this->_elems[get_index(i, check)];
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    virtual size_t compare(const GenericGrid1d<T,Mapfn>& ref, T epsilon,
                           int maxPrint = 0,
                           std::ostream& os = std::cerr) const {

        // Quick check for errors.
        size_t errs = count_diffs(ref, epsilon);

        // Run detailed comparison if any errors found.
        if (errs > 0 && maxPrint) {
            int p = 0;
            for (int i1 = -_p1; i1 < _d1 + _p1; i1++) {
                T te = (*this)(i1);
                T re = ref(i1);
                T err = te - re;
                if (err > epsilon || err < -epsilon) {
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
// Supports padding on each dimension.
// The Mapfn class must provide a 1:1 mapping between
// 2D and 1D indices.
template <typename T, typename Mapfn> class GenericGrid2d :
    public GenericGridBase<T> {
protected:
    const Mapfn _mapfn;
    const int _d1, _d2;
    const int _p1, _p2;
    
public:

    // Construct a grid of dimensions
    // d1+2*p1 x d2+2*p2,
    // i.e., with padding.
    GenericGrid2d(int d1, int d2,
                  int p1, int p2,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGridBase<T>((d1 + 2*p1) * 
                    (d2 + 2*p2), alignment),
        _mapfn(d1 + 2*p1,
               d2 + 2*p2),
        _d1(d1), _d2(d2),
        _p1(p1), _p2(p2) { }

    // Construct a grid of dimensions
    // d1 x d2, i.e., no padding.
    GenericGrid2d(int d1, int d2,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGrid2d(d1, 0, d2, 0, alignment) { } 

    // Get original parameters.
    inline int get_d1() const { return _d1; }
    inline int get_d2() const { return _d2; }
    inline int get_p1() const { return _p1; }
    inline int get_p2() const { return _p2; }

    // Get 1D index.
    // Thus, -p1 <= i < d1+p1, etc.
    inline size_t get_index(int i, int j, bool check=true) const {

        // Adjust for padding.
        int ip = i + _p1;
        int jp = j + _p2;
        if (check) {
            assert(ip >= 0);
            assert(ip < _mapfn.get_d1());
            assert(jp >= 0);
            assert(jp < _mapfn.get_d2());
        }
        size_t ai = _mapfn.map(ip, jp);
        if (check)
            assert(ai < _mapfn.get_size());
        return ai;
    }

    // Access element given 2D indices.
    inline const T& operator()(int i, int j, bool check=true) const {
        return this->_elems[get_index(i, j, check)];
    }

    // Non-const version.
    inline T& operator()(int i, int j, bool check=true) {
        return this->_elems[get_index(i, j, check)];
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    virtual size_t compare(const GenericGrid2d<T,Mapfn>& ref,
                           T epsilon,
                           int maxPrint = 0,
                           std::ostream& os = std::cerr) const {

        // Quick check for errors.
        size_t errs = count_diffs(ref, epsilon);

        // Run detailed comparison if any errors found.
        if (errs > 0 && maxPrint) {
            int p = 0;
            for (int i1 = -_p1; i1 < _d1 + _p1; i1++) {
                for (int i2 = -_p2; i2 < _d2 + _p2; i2++) {
                    T te = (*this)(i1, i2);
                    T re = ref(i1, i2);
                    T err = te - re;
                    if (err > epsilon || err < -epsilon) {
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
// Supports padding on each dimension.
// The Mapfn class must provide a 1:1 mapping between
// 3D and 1D indices.
template <typename T, typename Mapfn> class GenericGrid3d :
    public GenericGridBase<T> {
protected:
    Mapfn _mapfn;
    int _d1, _d2, _d3;
    int _p1, _p2, _p3;
    
public:

    // Construct a grid of dimensions
    // d1+2*p1 x d2+2*p2 x d3+2*p3,
    // i.e., with padding.
    GenericGrid3d(int d1, int d2, int d3,
                  int p1, int p2, int p3,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGridBase<T>((d1 + 2*p1) * 
                    (d2 + 2*p2) *
                    (d3 + 2*p3), alignment),
        _mapfn(d1 + 2*p1,
               d2 + 2*p2,
               d3 + 2*p3),
        _d1(d1), _d2(d2), _d3(d3),
        _p1(p1), _p2(p2), _p3(p3) { }

    // Construct a grid of dimensions
    // d1 x d2 x d3, i.e., no padding.
    GenericGrid3d(int d1, int d2, int d3,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGrid3d(d1, 0, d2, 0, d3, 0, alignment) { } 

    // Get original parameters.
    inline int get_d1() const { return _d1; }
    inline int get_d2() const { return _d2; }
    inline int get_d3() const { return _d3; }
    inline int get_p1() const { return _p1; }
    inline int get_p2() const { return _p2; }
    inline int get_p3() const { return _p3; }

    // Get 1D index.
    // Thus, -p1 <= i < d1+p1, etc.
    inline size_t get_index(int i, int j, int k, bool check=true) const {

        // Adjust for padding.
        int ip = i + _p1;
        int jp = j + _p2;
        int kp = k + _p3;
        if (check) {
            assert(ip >= 0);
            assert(ip < _mapfn.get_d1());
            assert(jp >= 0);
            assert(jp < _mapfn.get_d2());
            assert(kp >= 0);
            assert(kp < _mapfn.get_d3());
        }
        size_t ai = _mapfn.map(ip, jp, kp);
        if (check)
            assert(ai < _mapfn.get_size());
        return ai;
    }

    // Access element given 3D indices.
    inline const T& operator()(int i, int j, int k, bool check=true) const {
        return this->_elems[get_index(i, j, k, check)];
    }

    // Non-const version.
    inline T& operator()(int i, int j, int k, bool check=true) {
        return this->_elems[get_index(i, j, k, check)];
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    virtual size_t compare(const GenericGrid3d<T,Mapfn>& ref,
                           T epsilon,
                           int maxPrint = 0,
                           std::ostream& os = std::cerr) const {

        // Quick check for errors.
        size_t errs = count_diffs(ref, epsilon);

        // Run detailed comparison if any errors found.
        if (errs > 0 && maxPrint) {
            int p = 0;
            for (int i1 = -_p1; i1 < _d1 + _p1; i1++) {
                for (int i2 = -_p2; i2 < _d2 + _p2; i2++) {
                    for (int i3 = -_p3; i3 < _d3 + _p3; i3++) {
                        T te = (*this)(i1, i2, i3);
                        T re = ref(i1, i2, i3);
                        T err = te - re;
                        if (err > epsilon || err < -epsilon) {
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
// Supports padding on each dimension.
// The Mapfn class must provide a 1:1 mapping between
// 4D and 1D indices.
template <typename T, typename Mapfn> class GenericGrid4d :
    public GenericGridBase<T> {
protected:
    const Mapfn _mapfn;
    const int _d1, _d2, _d3, _d4;
    const int _p1, _p2, _p3, _p4;
    
public:

    // Construct a grid of dimensions
    // d1+2*p1 x d2+2*p2 x d3+2*p3 x d4+2*p4,
    // i.e., with padding.
    GenericGrid4d(int d1, int d2, int d3, int d4,
                  int p1, int p2, int p3, int p4,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGridBase<T>((d1 + 2*p1) * 
                           (d2 + 2*p2) *
                           (d3 + 2*p3) *
                           (d4 + 2*p4), alignment),
        _mapfn(d1 + 2*p1,
               d2 + 2*p2,
               d3 + 2*p3,
               d4 + 2*p4),
        _d1(d1), _d2(d2), _d3(d3), _d4(d4),
        _p1(p1), _p2(p2), _p3(p3), _p4(p4) { }

    // Construct a grid of dimensions
    // d1 x d2 x d3 x d4, i.e., no padding.
    GenericGrid4d(int d1, int d2, int d3, int d4,
                  size_t alignment=GenericGridBase<T>::_def_alignment) :
        GenericGrid4d(d1, 0, d2, 0, d3, 0, d4, 0, alignment) { } 

    // Get original parameters.
    inline int get_d1() const { return _d1; }
    inline int get_d2() const { return _d2; }
    inline int get_d3() const { return _d3; }
    inline int get_d4() const { return _d4; }
    inline int get_p1() const { return _p1; }
    inline int get_p2() const { return _p2; }
    inline int get_p3() const { return _p3; }
    inline int get_p4() const { return _p4; }

    // Get 1D index.
    // Thus, -p1 <= i < d1+p1, etc.
    inline size_t get_index(int i, int j, int k, int l, bool check=true) const {

        // Adjust for padding.
        int ip = i + _p1;
        int jp = j + _p2;
        int kp = k + _p3;
        int lp = l + _p4;
        if (check) {
            assert(ip >= 0);
            assert(ip < _mapfn.get_d1());
            assert(jp >= 0);
            assert(jp < _mapfn.get_d2());
            assert(kp >= 0);
            assert(kp < _mapfn.get_d3());
            assert(lp >= 0);
            assert(lp < _mapfn.get_d4());
        }
        size_t ai = _mapfn.map(ip, jp, kp, lp);
        if (check)
            assert(ai < _mapfn.get_size());
        return ai;
    }

    // Access element given 4D indices.
    inline const T& operator()(int i, int j, int k, int l, bool check=true) const {
        return this->_elems[get_index(i, j, k, l, check)];
    }

    // Non-const version.
    inline T& operator()(int i, int j, int k, int l, bool check=true) {
        return this->_elems[get_index(i, j, k, l, check)];
    }

    // Check for equality.
    // Return number of mismatches greater than epsilon.
    virtual size_t compare(const GenericGrid4d<T,Mapfn>& ref,
                           T epsilon,
                           int maxPrint = 0,
                           std::ostream& os = std::cerr) const {

        // Quick check for errors.
        size_t errs = count_diffs(ref, epsilon);

        // Run detailed comparison if any errors found.
        if (errs > 0 && maxPrint) {
            int p = 0;
            for (int i1 = -_p1; i1 < _d1 + _p1; i1++) {
                for (int i2 = -_p2; i2 < _d2 + _p2; i2++) {
                    for (int i3 = -_p3; i3 < _d3 + _p3; i3++) {
                        for (int i4 = -_p4; i4 < _d4 + _p4; i4++) {
                            T te = (*this)(i1, i2, i3, i4);
                            T re = ref(i1, i2, i3, i4);
                            T err = te - re;
                            if (err > epsilon || err < -epsilon) {
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

#endif
