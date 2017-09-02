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

    // A base class for a generic n-D grid.
    // This class does not define a type or memory layout.
    class GenericGridBase {

    protected:
        std::string _name;      // name for grid.

        std::shared_ptr<char> _base; // base address of malloc'd memory.

        // Note that both _dims and *_layout_base hold dimensions unless this
        // is a scalar. For a scalar, _dims is empty and _layout_base = 0.
        IdxTuple _dims;         // names and lengths of dimensions.
        Layout* _layout_base = 0; // memory layout.

        // Output stream for messages.
        // Pointer-to-pointer to let it follow a parent's pointer.
        std::ostream** _ostr = 0;

        void _sync_dims_with_layout() {
            Indices idxs = _layout_base->get_sizes();
            idxs.setTupleVals(_dims);
        }
        void _sync_layout_with_dims() {
            Indices idxs(_dims);
            _layout_base->set_sizes(idxs);
        }
        
    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridBase(std::string name,
                        Layout& layout_base,
                        const GridDimNames& dimNames,
                        std::ostream** ostr) :
            _name(name), _layout_base(&layout_base), _ostr(ostr) {
            for (auto& dn : dimNames)
                _dims.addDimBack(dn, 1);
            _sync_layout_with_dims();
        }

        virtual ~GenericGridBase() { }

        // Perform default allocation.
        virtual void default_alloc() =0;

        // Access name.
        const std::string& get_name() const { return _name; }
        void set_name(const std::string& name) { _name = name; }

        // Access dims.
        const IdxTuple& get_dims() const { return _dims; }

        // Get the messsage output stream.
        virtual std::ostream& get_ostr() const {
            assert(_ostr);
            assert(*_ostr);
            return **_ostr;
        }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _dims.product();
        }

        // Get size in bytes.
        virtual size_t get_num_bytes() const =0;

        // Get number of dimensions.
        virtual int get_num_dims() const {
            return _dims.getNumDims();
        }

        // Get the nth dim name.
        virtual const std::string& get_dim_name(int n) const {
            return _dims.getDimName(n);
        }

        // Is dim used?
        virtual bool is_dim_used(const std::string& dim) const {
            return _dims.lookup(dim) != 0;
        }

        // Access nth dim size.
        idx_t get_dim_size(int n) const {
            return _dims.getVal(n);
        }
        void set_dim_size(int n, idx_t size) {
            _dims.setVal(n, size);
            _sync_layout_with_dims();
        }

        // Access all dim sizes.
        virtual Indices get_dim_sizes() const {
            return _layout_base->get_sizes();
        }
        void set_dim_sizes(const Indices& sizes) {
            for (int i = 0; i < _dims.size(); i++)
                _dims.setVal(i, sizes[i]);
            _sync_layout_with_dims();
        }

        // Return 'true' if dimensions are same names
        // and sizes, 'false' otherwise.
        inline bool are_dims_and_sizes_same(const GenericGridBase& src) {
            return _dims == src._dims;
        }

        // Print some descriptive info.
        virtual void print_info(std::ostream& os,
                                const std::string& elem_name) const =0;

        // Get 1D index.
        // Should be overridden by derived classes for efficiency.
        virtual idx_t get_index(const Indices& idxs, bool check=true) const {
            if (check) {
                for (int i = 0; i < _dims.size(); i++) {
                    idx_t j = idxs[i];
                    assert(j >= 0);
                    assert(j < _dims.getVal(i));
                }
            }
            idx_t ai = _layout_base->layout(idxs);
            if (check)
                assert(ai < get_num_elems());
            return ai;
        }
        virtual idx_t get_index(const IdxTuple& pt, bool check=true) const {
            assert(_dims.areDimsSame(pt));
            Indices idxs(pt);
            return get_index(idxs, check);
        }

        // Direct access to data.
        virtual void* get_storage() =0;
        virtual const void* get_storage() const =0;

        // Release storage.
        virtual void release_storage() =0;
        
        // Set pointer to storage.
        // Free old storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        virtual void set_storage(std::shared_ptr<char>& base, size_t offset) =0;

        // Check for equality, assuming same layout.
        // Return number of mismatches greater than epsilon.
        virtual idx_t count_diffs(const GenericGridBase* ref,
                                  double epsilon) const =0;

        // Check for equality.
        // Return number of mismatches greater than epsilon up to 'maxPrint'+1.
        virtual idx_t compare(const GenericGridBase* ref,
                              double epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const =0;
    };
    
    // A base class for a generic n-D grid of elements of arithmetic type T.
    // This class defines the type but does not define the memory layout.
    template <typename T>
    class GenericGridTemplate : public GenericGridBase {

    protected:
        T* _elems = 0;          // actual data, which may be offset from _base.

    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridTemplate(std::string name,
                            Layout& layout_base,
                            const GridDimNames& dimNames,
                            std::ostream** ostr) :
            GenericGridBase(name, layout_base, dimNames, ostr) { }

        // Dealloc _base when last pointer to it is destructed.
        virtual ~GenericGridTemplate() {

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
        
        // Get size in bytes.
        virtual size_t get_num_bytes() const {
            return sizeof(T) * get_num_elems();
        }

        // Print some descriptive info to 'os'.
        virtual void print_info(std::ostream& os,
                                const std::string& elem_name) const {
            if (_dims.getNumDims() == 0)
                os << "scalar";
            else
                os << _dims.getNumDims() << "-D grid (" <<
                    _dims.makeDimValStr(" * ") << ")";
            os << " '" << _name << "'";
            if (_elems)
                os << " with data at " << _elems << " containing ";
            else
                os << " with data not allocated for ";
            os << makeNumStr(get_num_elems()) << " " <<
                    elem_name << " element(s) of " <<
                    sizeof(T) << " byte(s) each = " <<
                    makeByteStr(get_num_bytes());
        }

        // Initialize all elements to the same given value.
        virtual void set_elems_same(T val) {
            if (_elems) {

#pragma omp parallel for
                for (idx_t ai = 0; ai < get_num_elems(); ai++)
                    _elems[ai] = val;
            }
        }

        // Initialize memory using 'seed' as a starting point.
        virtual void set_elems_in_seq(T seed) {
            if (_elems) {
                const idx_t wrap = 71; // prime number is good to use.

#pragma omp parallel for
                for (idx_t ai = 0; ai < get_num_elems(); ai++)
                    _elems[ai] = seed * T(ai % wrap + 1);
            }
        }

        // Return given element.
        const T& operator()(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }
        const T& operator()(const IdxTuple& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }

        // Non-const access to given element.
        T& operator()(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }
        T& operator()(const IdxTuple& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }
        
        // Direct access to data.
        virtual void* get_storage() {
            return (void*)_elems;
        }
        virtual const void* get_storage() const {
            return (void*)_elems;
        }

        // Release storage.
        virtual void release_storage() {
            _base.reset();
            _elems = 0;
        }
        
        // Set pointer to storage.
        // Free old storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        virtual void set_storage(std::shared_ptr<char>& base, size_t offset) {

            // Release any old data if last owner.
            release_storage();
            
            // Share ownership of base.
            // This ensures that last grid to use a shared allocation
            // will trigger dealloc.
            _base = base;
            
            // Set plain pointer to new data.
            if (base.get()) {
                char* p = _base.get() + offset;
                _elems = (T*)p;
            } else {
                _elems = 0;
            }
        }

        // Check for equality, assuming same layout.
        // Return number of mismatches greater than epsilon.
        virtual idx_t count_diffs(const GenericGridBase* ref,
                                  double epsilon) const {

            if (!ref)
                return get_num_elems();
            auto* p = dynamic_cast<const GenericGridTemplate<T>*>(ref);
            if (!p)
                return get_num_elems();

            // Dims & sizes same?
            if (_dims != p->_dims)
                return get_num_elems();

            // Count abs diffs > epsilon.
            T ep = epsilon;
            idx_t errs = 0;
#pragma omp parallel for reduction(+:errs)
            for (idx_t ai = 0; ai < get_num_elems(); ai++) {
                if (!within_tolerance(_elems[ai], p->_elems[ai], ep))
                    errs++;
            }

            return errs;
        }

        // Check for equality.
        // Return number of mismatches greater than epsilon up to 'maxPrint'+1.
        virtual idx_t compare(const GenericGridBase* ref,
                              double epsilon,
                              int maxPrint = 0,
                              std::ostream& os = std::cerr) const {

            if (!ref)
                return get_num_elems();
            auto* p = dynamic_cast<const GenericGridTemplate<T>*>(ref);
            if (!p)
                return get_num_elems();
            
            // Dims & sizes same?
            if (_dims != p->_dims)
                return get_num_elems();

            // Quick check for errors, assuming same layout.
            idx_t errs = count_diffs(p, epsilon);
            if (!errs)
                return 0;

            // Run detailed comparison if any errors found.
            errs = 0;
            T ep = epsilon;
            _dims.visitAllPoints([&](const IdxTuple& pt){
                    auto& te = (*this)(pt);
                    auto& re = (*p)(pt);
                    if (!within_tolerance(te, re, ep)) {
                        errs++;
                        if (errs < maxPrint)
                                os << "** mismatch at (" << pt.makeDimValStr() << "): " <<
                                    te << " != " << re << std::endl;
                        else if (errs == maxPrint)
                            os << "** Additional errors not printed." << std::endl;
                        else {
                            // errs > maxPrint.
                            return false;
                        }
                    }
                    return true;
                });
            return errs;
        }
    };

    // A generic n-D grid of elements of type T.
    // This class defines the type and memory layout.
    // The LayoutFn class must provide a 1:1 transform between
    // n-D and 1-D indices.
    template <typename T, typename LayoutFn>
    class GenericGrid :
        public GenericGridTemplate<T> {
    protected:
        LayoutFn _layout;
    
    public:

        // Construct an unallocated grid.
        GenericGrid(std::string name,
                    const GridDimNames& dimNames,
                    std::ostream** ostr) :
            GenericGridTemplate<T>(name, _layout, dimNames, ostr) {
            assert(dimNames.size() == _layout.get_num_sizes());
        }

        // Get number of dims.
        // More efficient version overriding base method because layout is known.
        virtual int get_num_dims() const final {
            return _layout.get_num_sizes();
        }

        // Get sizes of dims.
        // More efficient version overriding base method because layout is known.
        virtual Indices get_dim_sizes() const final {
            return _layout.get_sizes();
        }
        
        // Get 1D index.
        // More efficient version overriding base method because layout is known.
        virtual idx_t get_index(const Indices& idxs, bool check=true) const final {
#ifdef DEBUG
            return GenericGridTemplate<T>::get_index(idxs, check);
#else
            idx_t ai = _layout.layout(idxs);
            return ai;
#endif
        }
    };

    // A generic 0-D grid (scalar) of elements of type T.
    // Special case: No layout or dim names needed.
    template <typename T> class GenericScalar :
        public GenericGrid<T, Layout_0d> {

    protected:

        // List of dims is for consistency; should be empty for 0-D.
        const GridDimNames _dimNames;
    
    public:

        // Construct an unallocated scalar.
        GenericScalar(std::string name) :
            GenericGrid<T, Layout_0d>(name, _dimNames) { }
    };

}

#if GRID_TEST
    // Dummy vars to test template compiles.
    GridDimNames dummyDims;
    GenericGrid<idx_t, Layout_123> dummy1("d1", dummyDims);
    GenericGrid<real_t, Layout_123> dummy2("d2", dummyDims);
    GenericGrid<real_vec_t, Layout_123> dummy3("d3", dummyDims);
#endif
    
#endif
