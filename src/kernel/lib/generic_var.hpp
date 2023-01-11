/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2023, Intel Corporation

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

    // Forward decls.
    class GenericVarBase;
    template <typename T>
    class GenericVarTyped;
    template <typename T, typename LayoutFn>
    class GenericVar;
    
    // Core elements of a generic n-D var of elements of type T.
    // This class defines the type and memory layout.
    // The LayoutFn class must provide a 1:1 transform between
    // n-D and 1-D indices with a constant stride between
    // consecutively-indexed elements in each dim.
    // A trivially-copyable type for offloading.
    template <typename T, typename LayoutFn>
    class GenericVarCore {
        friend class GenericVarBase;
        friend class GenericVarTyped<T>;
        friend class GenericVar<T, LayoutFn>;

    protected:
        // Sizes and index transform functions.
        // Sizes are copies of GenericVarBase::_var_dims.
        LayoutFn _layout;
        static_assert(std::is_trivially_copyable<LayoutFn>::value,
                      "Needed for OpenMP offload");

        // Start of actual data, which may be offset from GenericVarBase::_base.
        synced_ptr<T> _elems = 0;

    public:

        // Get number of dims.
        ALWAYS_INLINE idx_t get_num_dims() const {
            return _layout.get_num_sizes();
        }
 
        // Get 1D index using layout.
        // Basically a wrapper around _layout.layout(), but with range checking for debug.
        ALWAYS_INLINE idx_t get_index(const Indices& idxs, bool check=true) const {
            #ifdef CHECK
            if (check) {

                // Make sure all indices are in bounds.
                for (int i = 0; i < _layout.get_num_sizes(); i++) {
                    idx_t j = idxs[i];
                    host_assert(j >= 0);
                    host_assert(j < _layout.get_size(i));
                }
            }

            // Strictly, _elems doesn't need to be valid when 'get_index()' is called
            // because we're not accessing data. But we will make this restriction
            // when 'check' is 'true'.
            if (check)
                host_assert(_elems.get());
            #endif

            idx_t ai = _layout.layout(idxs);

            #ifdef CHECK
            if (check) {

                // Make sure final 1D index is in bounds.
                host_assert(ai >= 0);
                host_assert(ai < _layout.get_num_elements());
            }
            #endif
            return ai;
        }

        // Return pointer to given element.
        ALWAYS_INLINE const T* get_ptr(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return &_elems[ai];
        }
        ALWAYS_INLINE T* get_ptr(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return &_elems[ai];
        }

        // Return ref to given element.
        ALWAYS_INLINE const T& operator()(const Indices& pt, bool check=true) const {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }
        ALWAYS_INLINE T& operator()(const Indices& pt, bool check=true) {
            idx_t ai = get_index(pt, check);
            return _elems[ai];
        }

    }; //GenericVarCore.
   
    // A base class for a generic n-D var.
    // This class does not define a type or memory layout.
    // This class is pure virtual.
    class GenericVarBase :
        public KernelStateBase {

    protected:

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

        // Names and sizes of dims in this var.
        IdxTuple _var_dims;

        // Ctor. No allocation is done. See notes on default_alloc().
        // This is protected to avoid construction except by derived type.
        GenericVarBase(KernelStateBase& state,
                       const std::string& name,
                       const VarDimNames& dim_names);
        // Dtor.
        virtual ~GenericVarBase() { }

    public:

        // Access name.
        const std::string& get_name() const { return _name; }
        void set_name(const std::string& name) { _name = name; }

        // NUMA accessors.
        int get_numa_pref() const {
            STATE_VARS_CONST(this);
            return (_numa_pref != _numa_unset) ?
                _numa_pref : actl_opts->_numa_pref;
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
            return _var_dims.get_num_dims();
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

        // Modify dim sizes.
        virtual void set_dim_size(int n, idx_t size) =0;
        virtual void set_dim_sizes(const Indices& sizes) =0;
        
        // Return 'true' if dimensions are same names
        // and sizes, 'false' otherwise.
        bool are_dims_and_sizes_same(const GenericVarBase& src) const {
            return _var_dims == src._var_dims;
        }

        // Direct access to data.
        virtual const void* get_storage() const =0;
        virtual void* get_storage() =0;

        // Free any old storage.
        // Set pointer to storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        virtual void set_storage(std::shared_ptr<char>& base, size_t offset) =0;

        // Release storage.
        virtual void release_storage(bool reset_ptr) =0;

        // Perform default allocation.
        // For other options,
        // programmer should call get_num_elems() or get_num_bytes() and
        // then provide allocated memory via set_storage().
        virtual void default_alloc() =0;

        // Get size in bytes.
        virtual size_t get_num_bytes() const =0;
    };

    // A base class for a generic n-D var of elements of arithmetic type T.
    // This class defines the type but does not define the memory layout.
    // This class is pure virtual because its base is pure virtual.
    template <typename T>
    class GenericVarTyped : public GenericVarBase {

    protected:

        // Ctor. No allocation is done. See notes on default_alloc().
        // This is protected to avoid construction except by derived type.
        GenericVarTyped(KernelStateBase& state,
                        const std::string& name,
                        const VarDimNames& dim_names) :
            GenericVarBase(state, name, dim_names) { }

        // Direct access to storage ptr.
        virtual synced_ptr<T>& get_elems() =0;

    public:

        // Get size of one element.
        size_t get_elem_bytes() const {
            return sizeof(T);
        }

        // Get size in bytes.
        size_t get_num_bytes() const override {
            return sizeof(T) * get_num_elems();
        }

        // Free any old storage.
        // Set pointer to storage.
        // 'base' should provide get_num_bytes() bytes at offset bytes.
        void set_storage(std::shared_ptr<char>& base, size_t offset) override;

        // Release storage.
        void release_storage(bool reset_ptr) override;

        // Sync pointer to data.
        void sync_data_ptr() {
            get_elems().sync();
        }

        // Perform default allocation.
        void default_alloc() override;

        // Print some descriptive info.
        std::string make_info_string(const std::string& elem_name) const;

        // Initialize all elements to the same given value.
        void set_elems_same(T val);

        // Initialize memory using 'seed' as a starting point.
        void set_elems_in_seq(T seed);
    };

    // A generic n-D var of elements of type T.
    // A pointer to a GenericVarCore obj must be given at construction.
    // The GenericVar does NOT own the GenericVarCore obj.
    template <typename T, typename LayoutFn>
    class GenericVar :
        public GenericVarTyped<T> {

    protected:
        typedef GenericVarCore<T, LayoutFn> _core_t;
        _core_t* _corep;
        static_assert(std::is_trivially_copyable<_core_t>::value,
                      "Needed for OpenMP offload");

        // Both _var_dims and _core._layout hold sizes unless this is a
        // scalar. (For a scalar, _var_dims is empty.)
        // These functions keep them in sync.
        void _sync_dims_with_layout() {
            Indices idxs(_corep->_layout.get_sizes());
            idxs.set_tuple_vals(GenericVarBase::_var_dims);
        }
        void _sync_layout_with_dims() {
            STATE_VARS(this);
            Indices idxs(GenericVarBase::_var_dims);
            _corep->_layout.set_sizes(idxs);
        }

        // Direct access to storage ptr.
        // Allows modifying the pointer itself.
        synced_ptr<T>& get_elems() override {
            return _corep->_elems;
        }

    public:

        // Construct an unallocated var.
        // Must supply a pointer to an existing _core_t.
        GenericVar(KernelStateBase& state,
                   _core_t* corep,
                   std::string name,
                   const VarDimNames& dim_names) :
            GenericVarTyped<T>(state, name, dim_names),
            _corep(corep) {
            assert(_corep);

            // '_var_dims' was set in GenericVarBase construction.
            // Need to sync '_layout' w/it.
            _sync_layout_with_dims();
            assert(int(dim_names.size()) == _corep->_layout.get_num_sizes());
        }

        ~GenericVar() {

            // Release data.
            GenericVarTyped<T>::release_storage(false);
        }

        // Direct access to data.
        const void* get_storage() const override {
            return (void*)_corep->_elems;
        }
        void* get_storage() override {
            return (void*)_corep->_elems;
        }

        // Modify dim sizes.
        void set_dim_size(int n, idx_t size) override {
            GenericVarBase::_var_dims.set_val(n, size);
            _sync_layout_with_dims();
        }
        void set_dim_sizes(const Indices& sizes) override {
            auto& vd = GenericVarBase::_var_dims;
            for (int i = 0; size_t(i) < vd.size(); i++)
                vd.set_val(i, sizes[i]);
            _sync_layout_with_dims();
        }

        // Access all dim sizes.
        inline const Indices& get_dim_sizes() const {
            return _corep->_layout.get_sizes();
        }

        // Get 1D index using layout.
        ALWAYS_INLINE idx_t get_index(const Indices& idxs, bool check=true) const {
            return _corep->get_index(idxs, check);
        }
        ALWAYS_INLINE idx_t get_index(const IdxTuple& pt, bool check=true) const {
            host_assert(GenericVarBase::_var_dims.are_dims_same(pt));
            Indices idxs(pt);
            return get_index(idxs, check);
        }

        // Pointer to given element.
        ALWAYS_INLINE const T* get_ptr(const Indices& pt, bool check=true) const {
            return _corep->get_ptr(pt, check);
        }
        ALWAYS_INLINE T* get_ptr(const Indices& pt, bool check=true) {
            return _corep->get_ptr(pt, check);
        }

        // Return ref to given element.
        ALWAYS_INLINE const T& operator()(const Indices& pt, bool check=true) const {
            return _corep->get_ptr(pt, check);
        }
        ALWAYS_INLINE const T& operator()(const IdxTuple& pt, bool check=true) const {
            return _corep->get_ptr(pt, check);
        }
        ALWAYS_INLINE T& operator()(const Indices& pt, bool check=true) {
            return _corep->get_ptr(pt, check);
        }
        ALWAYS_INLINE T& operator()(const IdxTuple& pt, bool check=true) {
            return _corep->get_ptr(pt, check);
        }

        // Compute strides.
        Indices get_strides() const {
            auto nd = _corep->get_num_dims();
            Indices strides(nd);
            for (int d = 0; d < nd; d++) {

                // For dim 'd', measure distance from index 0 to 1.
                auto idxs = Indices(idx_t(0), nd);
                auto i0 = get_index(idxs, false);
                idxs[d] = 1;
                auto i1 = get_index(idxs, false);
                auto sd = i1 - i0;
                strides[d] = sd;
                assert(sd >= 0);

                // Check that the distance holds for other indices.
                #ifdef CHECK
                for (idx_t j : { 13, -17 }) {
                    idxs[d] = j;
                    auto i = get_index(idxs, false);
                    assert(i - i0 == sd * j);
                }
                #endif
            }
            return strides;
        }        
    };

} // namespace yask.
