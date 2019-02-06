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

// Generic grids:
// T: type stored in grid.
// LayoutFn: class that transforms N dimensions to 1.

#pragma once

namespace yask {

    // A base class for a generic n-D grid.
    // This class does not define a type or memory layout.
    class GenericGridBase :
        public KernelStateBase {

    protected:
        std::string _name;      // name for grid.

        // Base address of malloc'd memory.
        // Not necessarily the address at which the data is stored.
        // Several grids may have the same base addr; then the last one
        // to release it will free the mem.
        std::shared_ptr<char> _base;

        void* _elems = 0;          // actual data, which may be offset from _base.

        // Preferred NUMA node.
        const static int _numa_unset = -999;
        int _numa_pref = _numa_unset; // use default from _opts.

        // Note that both _dims and *_layout_base hold dimensions unless this
        // is a scalar. For a scalar, _dims is empty and _layout_base = 0.
        IdxTuple _grid_dims;         // names and lengths of grid dimensions.
        Layout* _layout_base = 0; // memory layout.

        void _sync_dims_with_layout() {
            Indices idxs(_layout_base->get_sizes());
            idxs.setTupleVals(_grid_dims);
        }
        void _sync_layout_with_dims() {
            Indices idxs(_grid_dims);
            _layout_base->set_sizes(idxs);
        }

    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridBase(KernelStateBase& state,
                        const std::string& name,
                        Layout& layout_base,
                        const GridDimNames& dimNames);

        virtual ~GenericGridBase() { }

        // Get state info.
        KernelStatePtr& get_state() {
            assert(_state);
            return _state;
        }
        const KernelStatePtr& get_state() const {
            assert(_state);
            return _state;
        }
        std::ostream& get_ostr() const {
            STATE_VARS(this);
            return os;
        }

        // Perform default allocation.
        // For other options,
        // programmer should call get_num_elems() or get_num_bytes() and
        // then provide allocated memory via set_storage().
        virtual void default_alloc();

        // Access name.
        const std::string& get_name() const { return _name; }
        void set_name(const std::string& name) { _name = name; }

        // NUMA accessors.
        virtual int get_numa_pref() const {
            STATE_VARS_CONST(this);
            return (_numa_pref != _numa_unset) ?
                _numa_pref : opts->_numa_pref;
        }
        virtual bool set_numa_pref(int numa_node) {
#ifdef USE_NUMA
            _numa_pref = numa_node;
            return true;
#else
            _numa_pref = yask_numa_none;
            return numa_node == yask_numa_none;
#endif
        }

        // Access dims of this grid.
        const IdxTuple& get_dims() const { return _grid_dims; }

        // Get number of elements.
        virtual idx_t get_num_elems() const {
            return _grid_dims.product();
        }

        // Get size of one element.
        virtual size_t get_elem_bytes() const =0;

        // Get size in bytes.
        virtual size_t get_num_bytes() const =0;

        // Get number of dimensions.
        virtual int get_num_dims() const {
            return _grid_dims.getNumDims();
        }

        // Get the nth dim name.
        virtual const std::string& get_dim_name(int n) const {
            return _grid_dims.getDimName(n);
        }

        // Is dim used?
        virtual bool is_dim_used(const std::string& dim) const {
            return _grid_dims.lookup(dim) != 0;
        }

        // Access nth dim size.
        idx_t get_dim_size(int n) const {
            return _grid_dims.getVal(n);
        }
        void set_dim_size(int n, idx_t size) {
            _grid_dims.setVal(n, size);
            _sync_layout_with_dims();
        }

        // Access all dim sizes.
        virtual const Indices& get_dim_sizes() const {
            return _layout_base->get_sizes();
        }
        void set_dim_sizes(const Indices& sizes) {
            for (int i = 0; i < _grid_dims.size(); i++)
                _grid_dims.setVal(i, sizes[i]);
            _sync_layout_with_dims();
        }

        // Return 'true' if dimensions are same names
        // and sizes, 'false' otherwise.
        inline bool are_dims_and_sizes_same(const GenericGridBase& src) {
            return _grid_dims == src._grid_dims;
        }

        // Print some descriptive info.
        virtual std::string make_info_string(const std::string& elem_name) const;

        // Get linear index.
        virtual idx_t get_index(const Indices& idxs, bool check=true) const =0;
        virtual idx_t get_index(const IdxTuple& pt, bool check=true) const {
            assert(_grid_dims.areDimsSame(pt));
            Indices idxs(pt);
            return get_index(idxs, check);
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
        virtual void set_storage(std::shared_ptr<char>& base, size_t offset);

        // Check for equality, assuming same layout.
        // Return number of mismatches greater than epsilon.
        virtual idx_t count_diffs(const GenericGridBase* ref,
                                  double epsilon) const =0;

    };

    // A base class for a generic n-D grid of elements of arithmetic type T.
    // This class defines the type but does not define the memory layout.
    template <typename T>
    class GenericGridTemplate : public GenericGridBase {

    public:

        // Ctor. No allocation is done. See notes on default_alloc().
        GenericGridTemplate(KernelStateBase& state,
                            const std::string& name,
                            Layout& layout_base,
                            const GridDimNames& dimNames) :
            GenericGridBase(state, name, layout_base, dimNames) { }

        // Dealloc _base when last pointer to it is destructed.
        virtual ~GenericGridTemplate() {

            // Release data.
            release_storage();
        }

        // Get size of one element.
        virtual size_t get_elem_bytes() const {
            return sizeof(T);
        }

        // Get size in bytes.
        virtual size_t get_num_bytes() const {
            return sizeof(T) * get_num_elems();
        }

        // Initialize all elements to the same given value.
        virtual void set_elems_same(T val);

        // Initialize memory using 'seed' as a starting point.
        virtual void set_elems_in_seq(T seed);

        // Return ref to given element.
        const T& operator()(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return ((T*)_elems)[ai];
        }
        const T& operator()(const IdxTuple& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return ((T*)_elems)[ai];
        }

        // Non-const access to given element.
        T& operator()(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return ((T*)_elems)[ai];
        }
        T& operator()(const IdxTuple& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return ((T*)_elems)[ai];
        }

        // Check for equality, assuming same layout.
        // Return number of mismatches greater than epsilon.
        virtual idx_t count_diffs(const GenericGridBase* ref,
                                  double epsilon) const;
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
        GenericGrid(KernelStateBase& state,
                    std::string name,
                    const GridDimNames& dimNames) :
            GenericGridTemplate<T>(state, name, _layout, dimNames) {
            assert(int(dimNames.size()) == _layout.get_num_sizes());
        }

        // Get number of dims.
        // More efficient version overriding base method because layout is known.
        virtual int get_num_dims() const final {
            return _layout.get_num_sizes();
        }

        // Get sizes of dims.
        // More efficient version overriding base method because layout is known.
        virtual const Indices& get_dim_sizes() const final {
            return _layout.get_sizes();
        }

        // Get 1D index using layout.
        virtual idx_t get_index(const Indices& idxs, bool check=true) const final {
#ifdef CHECK
            if (check) {
                for (int i = 0; i < this->_grid_dims.size(); i++) {
                    idx_t j = idxs[i];
                    assert(j >= 0);
                    assert(j < this->_grid_dims.getVal(i));
                }
            }
#endif
            idx_t ai = _layout.layout(idxs);
#ifdef CHECK
            if (check)
                assert(ai < this->get_num_elems());
#endif
            return ai;
        }

        // Pointer to given element.
        // Calling this helps the compiler avoid virtual function calls.
        const T* getPtr(const Indices& pt, bool check=true) const {
            idx_t ai = GenericGrid::get_index(pt, check);
            return &((T*)this->_elems)[ai];
        }
        T* getPtr(const Indices& pt, bool check=true) {
            idx_t ai = GenericGrid::get_index(pt, check);
            return &((T*)this->_elems)[ai];
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
        GenericScalar(KernelStateBase& state,
                      std::string name) :
            GenericGrid<T, Layout_0d>(state, name, _dimNames) { }
    };

} // namespace yask.
