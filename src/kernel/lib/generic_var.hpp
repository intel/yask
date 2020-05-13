/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2020, Intel Corporation

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

// Generic vars:
// T: type stored in var.

#pragma once

namespace yask {

    // A base class for a generic n-D var.
    // This class does not define a type or memory layout.
    // This class hierarchy is not virtual.
    class GenericVarBase :
        public KernelStateBase {

    protected:
        // Start of actual data, which is some offset from _base on host.
        void* _elems = 0;

        // Name for var.
        std::string _name;

        // Base address of malloc'd memory.
        // Not necessarily the address at which the data is stored.
        // Several vars may have the same base addr; then the last one
        // to release it will free the mem.
        std::shared_ptr<char> _base;

        // Preferred NUMA node.
        const static int _numa_unset = -999;
        int _numa_pref = _numa_unset; // use default from _opts.

        // Names and lengths of var dimensions.
        IdxTuple _var_dims;

        // Ctor. No allocation is done. See notes on default_alloc().
        // This is protected to avoid construction except by derived type.
        GenericVarBase(KernelStateBase& state,
                       const std::string& name,
                       const VarDimNames& dim_names);

    public:

        // Access name.
        const std::string& get_name() const { return _name; }
        void set_name(const std::string& name) { _name = name; }

        // NUMA accessors.
        int get_numa_pref() const {
            STATE_VARS_CONST(this);
            return (_numa_pref != _numa_unset) ?
                _numa_pref : opts->_numa_pref;
        }
        bool set_numa_pref(int numa_node) {
            #ifdef USE_NUMA
            _numa_pref = numa_node;
            return true;
            #else
            _numa_pref = yask_numa_none;
            return numa_node == yask_numa_none;
            #endif
        }

        // Access dims of this var (not necessarily same as solution dims).
        const IdxTuple& get_dim_tuple() const {
            return _var_dims;
        }

        // Get number of elements.
        idx_t get_num_elems() const {
            return _var_dims.product();
        }

        // Get number of dimensions.
        int get_num_dims() const {
            return _var_dims._get_num_dims();
        }

        // Get the nth dim name.
        const std::string& get_dim_name(int n) const {
            return _var_dims.get_dim_name(n);
        }

        // Is dim used?
        bool is_dim_used(const std::string& dim) const {
            return _var_dims.lookup(dim) != 0;
        }

        // Access nth dim size.
        idx_t get_dim_size(int n) const {
            return _var_dims.get_val(n);
        }

        // Return 'true' if dimensions are same names
        // and sizes, 'false' otherwise.
        bool are_dims_and_sizes_same(const GenericVarBase& src) const {
            return _var_dims == src._var_dims;
        }

        // Direct access to data.
        void* get_storage() {
            return (void*)_elems;
        }
        const void* get_storage() const {
            return (void*)_elems;
        }
    };

    // A base class for a generic n-D var of elements of arithmetic type T.
    // This class defines the type but does not define the memory layout.
    template <typename T>
    class GenericVarTyped : public GenericVarBase {

    protected:

        // Ctor. No allocation is done. See notes on default_alloc().
        // This is protected to avoid construction except by derived type.
        GenericVarTyped(KernelStateBase& state,
                        const std::string& name,
                        const VarDimNames& dim_names) :
            GenericVarBase(state, name, dim_names) { }

    public:

        // Get size of one element.
        size_t get_elem_bytes() const {
            return sizeof(T);
        }

        // Get size in bytes.
        size_t get_num_bytes() const {
            return sizeof(T) * get_num_elems();
        }

        // Free any old storage.
        // Set pointer to storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        void set_storage(std::shared_ptr<char>& base, size_t offset);

        // Release storage.
        void release_storage();

        // Perform default allocation.
        // For other options,
        // programmer should call get_num_elems() or get_num_bytes() and
        // then provide allocated memory via set_storage().
        void default_alloc();

        // Print some descriptive info.
        std::string make_info_string(const std::string& elem_name) const;

        // Initialize all elements to the same given value.
        void set_elems_same(T val);

        // Initialize memory using 'seed' as a starting point.
        void set_elems_in_seq(T seed);
    };

    // A generic n-D var of elements of type T.
    // This class defines the type and memory layout.
    // The LayoutFn class must provide a 1:1 transform between
    // n-D and 1-D indices.
    template <typename T, typename LayoutFn>
    class GenericVar :
        public GenericVarTyped<T> {

    protected:

        // Sizes and index transform functions.
        LayoutFn _layout;

        // Both _var_dims and _layout hold sizes unless this is a
        // scalar. (For a scalar, _var_dims is empty.)
        // These functions keep them in sync.
        void _sync_dims_with_layout() {
            Indices idxs(_layout.get_sizes());
            idxs.set_tuple_vals(GenericVarBase::_var_dims);
        }
        void _sync_layout_with_dims() {
            STATE_VARS(this);
            Indices idxs(GenericVarBase::_var_dims);
            _layout.set_sizes(idxs);
        }

    public:

        // Construct an unallocated var.
        GenericVar(KernelStateBase& state,
                   std::string name,
                   const VarDimNames& dim_names) :
            GenericVarTyped<T>(state, name, dim_names) {

            // '_var_dims' was set in GenericVar construction.
            // Need to sync '_layout' w/it.
            _sync_layout_with_dims();
            assert(int(dim_names.size()) == _layout.get_num_sizes());
        }

        ~GenericVar() {

            // Release data.
            GenericVarTyped<T>::release_storage();
        }

        // Modify dim sizes.
        void set_dim_size(int n, idx_t size) {
            GenericVarBase::_var_dims.set_val(n, size);
            _sync_layout_with_dims();
        }
        void set_dim_sizes(const Indices& sizes) {
            auto& vd = GenericVarBase::_var_dims;
            for (int i = 0; size_t(i) < vd.size(); i++)
                vd.set_val(i, sizes[i]);
            _sync_layout_with_dims();
        }

        // Access all dim sizes.
        inline const Indices& get_dim_sizes() const {
            return _layout.get_sizes();
        }

        // Get 1D index using layout.
        ALWAYS_INLINE idx_t get_index(const Indices& idxs, bool check=true) const {
            #ifdef CHECK
            if (check) {
                for (int i = 0; size_t(i) < this->_var_dims.size(); i++) {
                    idx_t j = idxs[i];
                    assert(j >= 0);
                    assert(j < this->_var_dims.get_val(i));
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
        ALWAYS_INLINE idx_t get_index(const IdxTuple& pt, bool check=true) const {
            assert(GenericVarBase::_var_dims.are_dims_same(pt));
            Indices idxs(pt);
            return get_index(idxs, check);
        }

        // Pointer to given element.
        ALWAYS_INLINE const T* get_ptr(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return &((T*)GenericVarBase::_elems)[ai];
        }
        ALWAYS_INLINE T* get_ptr(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return &((T*)GenericVarBase::_elems)[ai];
        }

        // Return const ref to given element.
        ALWAYS_INLINE const T& operator()(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return ((T*)GenericVarBase::_elems)[ai];
        }
        ALWAYS_INLINE const T& operator()(const IdxTuple& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return ((T*)GenericVarBase::_elems)[ai];
        }

        // Non-const access to given element.
        ALWAYS_INLINE T& operator()(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return ((T*)GenericVarBase::_elems)[ai];
        }
        ALWAYS_INLINE T& operator()(const IdxTuple& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return ((T*)GenericVarBase::_elems)[ai];
        }

    };

} // namespace yask.
