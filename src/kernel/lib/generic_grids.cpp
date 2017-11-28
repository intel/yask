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

#include "yask.hpp"

namespace yask {

    // Ctor. No allocation is done. See notes on default_alloc().
    GenericGridBase::GenericGridBase(std::string name,
                                     Layout& layout_base,
                                     const GridDimNames& dimNames,
                                     std::ostream** ostr) :
        _name(name), _layout_base(&layout_base), _ostr(ostr) {
        for (auto& dn : dimNames)
            _dims.addDimBack(dn, 1);
        _sync_layout_with_dims();
    }

    // Perform default allocation.
    // For other options,
    // programmer should call get_num_elems() or get_num_bytes() and
    // then provide allocated memory via set_storage().
    void GenericGridBase::default_alloc() {
        
        // Release any old data if last owner.
        release_storage();
        
        // Alloc required number of bytes.
        size_t sz = get_num_bytes();
        _base = std::shared_ptr<char>(alignedAlloc(sz), AlignedDeleter());
        
        // No offset.
        _elems = _base.get();
    }

    // Print some descriptive info to 'os'.
    void GenericGridBase::print_info(std::ostream& os,
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
            os << " with data not yet allocated for ";
        os << makeByteStr(get_num_bytes()) << 
            " (" << makeNumStr(get_num_elems()) << " " <<
            elem_name << " element(s) of " <<
            get_elem_bytes() << " byte(s) each)";
    }
    
    // Set pointer to storage.
    // Free old storage.
    // 'base' should provide get_num_bytes() bytes at offset bytes.
    void GenericGridBase::set_storage(std::shared_ptr<char>& base, size_t offset) {

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

} // yask namespace.
