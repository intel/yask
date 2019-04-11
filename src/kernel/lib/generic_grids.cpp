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

#include "yask_stencil.hpp"
using namespace std;

namespace yask {

    // Ctor. No allocation is done. See notes on default_alloc().
    GenericGridBase::GenericGridBase(KernelStateBase& state,
                                     const string& name,
                                     Layout& layout_base,
                                     const GridDimNames& dimNames) :
        KernelStateBase(state),
        _name(name),
        _layout_base(&layout_base) {
        for (auto& dn : dimNames)
            _grid_dims.addDimBack(dn, 1);
        _sync_layout_with_dims();
    }

    // Perform default allocation.
    // For other options,
    // programmer should call get_num_elems() or get_num_bytes() and
    // then provide allocated memory via set_storage().
    void GenericGridBase::default_alloc() {
        STATE_VARS(this);

        // Release any old data if last owner.
        release_storage();

        // What node?
        int numa_pref = get_numa_pref();

        // Alloc required number of bytes.
        size_t sz = get_num_bytes();
        os << "Allocating " << makeByteStr(sz) <<
            " for grid '" << _name << "'";
#ifdef USE_NUMA
        if (numa_pref >= 0)
            os << " preferring NUMA node " << numa_pref;
        else
            os << " on local NUMA node";
#endif
        os << "...\n" << flush;
        _base = shared_numa_alloc<char>(sz, numa_pref);

        // No offset.
        _elems = _base.get();
    }

    // Make some descriptive info.
    string GenericGridBase::make_info_string(const string& elem_name) const {
        stringstream oss;
        oss << "'" << _name << "' ";
        if (_grid_dims.getNumDims() == 0)
            oss << "scalar";
        else
            oss << _grid_dims.getNumDims() << "-D var (" <<
                _grid_dims.makeDimValStr(" * ") << ")";
        if (_elems)
            oss << " with storage at " << _elems << " containing ";
        else
            oss << " with storage not yet allocated for ";
        oss << makeByteStr(get_num_bytes()) <<
            " (" << makeNumStr(get_num_elems()) << " " <<
            elem_name << " element(s) of " <<
            get_elem_bytes() << " byte(s) each)";
        return oss.str();
    }

    // Set pointer to storage.
    // Free old storage.
    // 'base' should provide get_num_bytes() bytes at offset bytes.
    void GenericGridBase::set_storage(shared_ptr<char>& base, size_t offset) {

        // Release any old data if last owner.
        release_storage();

        // Share ownership of base.
        // This ensures that last grid to use a shared allocation
        // will trigger dealloc.
        _base = base;

        // Set plain pointer to new data.
        if (base.get()) {
            char* p = _base.get() + offset;
            _elems = (void*)p;
        } else {
            _elems = 0;
        }
    }

    // Template implementations.

    template <typename T>
    void GenericGridTemplate<T>::set_elems_same(T val) {
        if (_elems) {
            yask_for(0, get_num_elems(), 1,
                     [&](idx_t start, idx_t stop, idx_t thread_num) {
                         ((T*)_elems)[start] = val;
                     });
        }
    }
    
    template <typename T>
    void GenericGridTemplate<T>::set_elems_in_seq(T seed) {
        if (_elems) {
            const idx_t wrap = 71; // TODO: avoid multiple of any dim size.
            yask_for(0, get_num_elems(), 1,
                     [&](idx_t start, idx_t stop, idx_t thread_num) {
                         ((T*)_elems)[start] = seed * T(start % wrap + 1);
                     });
        }
    }

    template <typename T>
    idx_t GenericGridTemplate<T>::count_diffs(const GenericGridBase* ref,
                                              double epsilon) const {

        if (!ref)
            return get_num_elems();
        auto* p = dynamic_cast<const GenericGridTemplate<T>*>(ref);
        if (!p)
            return get_num_elems();

        // Dims & sizes same?
        if (_grid_dims != p->_grid_dims)
            return get_num_elems();

        // Count abs diffs > epsilon.
        T ep = epsilon;
        idx_t errs = 0;
#pragma omp parallel for reduction(+:errs)
        for (idx_t ai = 0; ai < get_num_elems(); ai++) {
            if (!within_tolerance(((T*)_elems)[ai], ((T*)p->_elems)[ai], ep))
                errs++;
        }
        return errs;
    }

    // Explicitly allowed instantiations.
    template class GenericGridTemplate<real_t>;
    template class GenericGridTemplate<real_vec_t>;

} // yask namespace.
