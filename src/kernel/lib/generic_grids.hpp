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

#include "tuple.hpp"

namespace yask {

    typedef Tuple<idx_t> IdxTuple;
    typedef std::vector<idx_t> GridIndices;

    // A base class for a generic grid of elements of arithmetic type T.
    // This class provides linear-access support, i.e., no layout.
    template <typename T> class GenericGridBase {
    protected:
        std::string _name;      // name for grid.
        IdxTuple _dims;         // names and lengths of dimensions.
        std::shared_ptr<char> _base; // base address of malloc'd memory.
        T* _elems = 0;          // actual data, which may be offset from _base.

    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridBase(std::string name) :
            _name(name) { }

        // Dealloc _base when last pointer to it is destructed.
        virtual ~GenericGridBase() {

            // Release data.
            release_storage();
        }

        // Perform default allocation.
        // For other options,
        // programmer should call get_num_elems() or get_num_bytes() and
        // then provide allocated memory via set_storage().
        virtual void default_alloc() {

            // Release any old data if last owner.
            release_storage();

            // Alloc required number of bytes.
            size_t sz = get_num_bytes();
            _base = std::shared_ptr<char>(alignedAlloc(sz), AlignedDeleter());

            // No offset.
            _elems = (T*)_base.get();
        }
        
        // Access name.
        const std::string& get_name() const { return _name; }
        void set_name(const std::string& name) { _name = name; }

        // Get number of elements.
        virtual idx_t get_num_elems() const =0;

        // Get size in bytes.
        inline size_t get_num_bytes() const {
            return sizeof(T) * get_num_elems();
        }

        // Get number of dimensions.
        inline int get_num_dims() const {
            return _dims.getNumDims();
        }

        // Get the nth dim name.
        inline const std::string& get_dim_name(int n) const {
            return _dims.getDimName(n);
        }

        // Is dim used?
        virtual bool is_dim_used(const std::string& dim) const {
            return _dims.lookup(dim) != 0;
        }

        // Get the nth dim size.
        inline idx_t get_dim_size(int n) const {
            return _dims.getVal(n);
        }

        // Return 'true' if dimensions are same names
        // and sizes, 'false' otherwise.
        inline bool are_dims_same(const GenericGridBase& src) {
            return _dims == src._dims;
        }

        // Print some descriptive info to 'os'.
        virtual void print_info(std::ostream& os,
                                const std::string& elem_name) const {
            if (_dims.getNumDims() == 0)
                os << "scalar";
            else
                os << _dims.getNumDims() << "D grid (" <<
                    _dims.makeDimValStr(" * ") << ")";
            os << " '" << _name << "'";
            if (_elems)
                os << ", data at " << _elems << ", containing " <<
                    printWithPow10Multiplier(get_num_elems()) <<
                    elem_name << " element(s) of " <<
                    sizeof(T) << " byte(s) each, " <<
                    printWithPow2Multiplier(get_num_bytes()) << "B";
        }

        // Initialize all elements to the same given value.
        virtual void set_same(T val) {
            if (_elems) {

#pragma omp parallel for
                for (idx_t ai = 0; ai < get_num_elems(); ai++)
                    _elems[ai] = val;
            }
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
        // Assumes same layout.
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

        // Release storage.
        void release_storage() {
            _base.reset();
            _elems = 0;
        }
        
        // Set pointer to storage.
        // Free old storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        void set_storage(std::shared_ptr<char>& base, size_t offset) {

            // Release any old data if last owner.
            release_storage();
            
            // Share ownership of base.
            _base = base;
            
            // Set plain pointer to new data.
            if (base.get()) {
                char* p = _base.get() + offset;
                _elems = (T*)p;
            } else {
                _elems = 0;
            }
        }
    };

    // A generic 0D grid (scalar) of elements of type T.
    // No layout function needed, because there is only 1 element.
    template <typename T> class GenericGrid0d :
        public GenericGridBase<T> {
    
    public:

        // Construct an unallocated scalar.
        GenericGrid0d(std::string name) :
            GenericGridBase<T>(name) { }

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
        GenericGrid1d(std::string name,
                      const std::string& dim1) :
            GenericGridBase<T>(name) {
            this->_dims.addDimBack(dim1, 1);
        }

        // Get/set size.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline void set_d1(idx_t d1) {
            _layout.set_d1(d1);
            this->_dims.setVal(0, d1); }
        inline void set_dim_sizes(idx_t d1) {
            set_d1(d1);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
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
        GenericGrid2d(std::string name,
                      const std::string& dim1,
                      const std::string& dim2) :
            GenericGridBase<T>(name) {
            this->_dims.addDimBack(dim1, 1);
            this->_dims.addDimBack(dim2, 1);
        }
        
        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline void set_d1(idx_t d1) {
            _layout.set_d1(d1); 
            this->_dims.setVal(0, d1);
        }
        inline void set_d2(idx_t d2) {
            _layout.set_d2(d2); 
            this->_dims.setVal(0, d2);
        }
        inline void set_dim_sizes(idx_t d1, idx_t d2) {
            set_d1(d1);
            set_d2(d2);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
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
        GenericGrid3d(const std::string& name,
                      const std::string& dim1,
                      const std::string& dim2,
                      const std::string& dim3) :
            GenericGridBase<T>(name) {
            this->_dims.addDimBack(dim1, 1);
            this->_dims.addDimBack(dim2, 1);
            this->_dims.addDimBack(dim3, 1);
        }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline void set_d1(idx_t d1) {
            _layout.set_d1(d1);
            this->_dims.setVal(0, d1);
        }
        inline void set_d2(idx_t d2) {
            _layout.set_d2(d2);
            this->_dims.setVal(1, d2);
        }
        inline void set_d3(idx_t d3) {
            _layout.set_d3(d3);
            this->_dims.setVal(2, d3);
        }
        inline void set_dim_sizes(idx_t d1, idx_t d2, idx_t d3) {
            set_d1(d1);
            set_d2(d2);
            set_d3(d3);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
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
        GenericGrid4d(const std::string& name,
                      const std::string& dim1,
                      const std::string& dim2,
                      const std::string& dim3,
                      const std::string& dim4) :
            GenericGridBase<T>(name) {
            this->_dims.addDimBack(dim1, 1);
            this->_dims.addDimBack(dim2, 1);
            this->_dims.addDimBack(dim3, 1);
            this->_dims.addDimBack(dim4, 1);
        }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline idx_t get_d4() const { return _layout.get_d4(); }
        inline void set_d1(idx_t d1) {
            _layout.set_d1(d1);
            this->_dims.setVal(0, d1);
        }
        inline void set_d2(idx_t d2) {
            _layout.set_d2(d2);
            this->_dims.setVal(1, d2);
        }
        inline void set_d3(idx_t d3) {
            _layout.set_d3(d3);
            this->_dims.setVal(2, d3);
        }
        inline void set_d4(idx_t d4) {
            _layout.set_d4(d4);
            this->_dims.setVal(3, d4);
        }
        inline void set_dim_sizes(idx_t d1, idx_t d2, idx_t d3, idx_t d4) {
            set_d1(d1);
            set_d2(d2);
            set_d3(d3);
            set_d4(d4);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
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
        GenericGrid5d(const std::string& name,
                      const std::string& dim1,
                      const std::string& dim2,
                      const std::string& dim3,
                      const std::string& dim4,
                      const std::string& dim5) :
            GenericGridBase<T>(name) {
            this->_dims.addDimBack(dim1, 1);
            this->_dims.addDimBack(dim2, 1);
            this->_dims.addDimBack(dim3, 1);
            this->_dims.addDimBack(dim4, 1);
            this->_dims.addDimBack(dim5, 1);
        }

        // Get/set sizes.
        inline idx_t get_d1() const { return _layout.get_d1(); }
        inline idx_t get_d2() const { return _layout.get_d2(); }
        inline idx_t get_d3() const { return _layout.get_d3(); }
        inline idx_t get_d4() const { return _layout.get_d4(); }
        inline idx_t get_d5() const { return _layout.get_d5(); }
        inline void set_d1(idx_t d1) {
            _layout.set_d1(d1);
            this->_dims.setVal(0, d1);
        }
        inline void set_d2(idx_t d2) {
            _layout.set_d2(d2);
            this->_dims.setVal(1, d2);
        }
        inline void set_d3(idx_t d3) {
            _layout.set_d3(d3);
            this->_dims.setVal(2, d3);
        }
        inline void set_d4(idx_t d4) {
            _layout.set_d4(d4);
            this->_dims.setVal(3, d4);
        }
        inline void set_d5(idx_t d5) {
            _layout.set_d5(d5);
            this->_dims.setVal(4, d5);
        }
        inline void set_dim_sizes(idx_t d1, idx_t d2, idx_t d3, idx_t d4, idx_t d5) {
            set_d1(d1);
            set_d2(d2);
            set_d3(d3);
            set_d4(d4);
            set_d5(d5);
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _layout.get_size();
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
