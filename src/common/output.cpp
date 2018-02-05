/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

//////// Methods for output object. //////////

#include "yask_common_api.hpp"

// Common utilities.
#include "common_utils.hpp"

#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

namespace yask {

    class OutputBase : public virtual yask_output {
    protected:
        ostream& _os;
    public:
        OutputBase(ostream& os) : _os(os) {}
        
        virtual ostream& get_ostream() {
            return _os;
        }
    };

    class FileOutput : public OutputBase,
                       public virtual yask_file_output {
    protected:
        ofstream _ofs;
        string _fname;
    public:
        FileOutput() : OutputBase(_ofs) {}
        
        virtual ~FileOutput() {
            close();
        }
        
        string get_filename() const {
            return _fname;
        }
        
        void open(const string& file_name) {
            _fname = file_name;
            _ofs.open(file_name, ofstream::out | ofstream::trunc);
            if (!_ofs.is_open()) {
                THROW_YASK_EXCEPTION("Error: cannot open '" << file_name <<
                    "' for output.\n");
            }
        }

        void close() {
            if (_ofs.is_open())
                _ofs.close();
        }
    };

    class StringOutput : public OutputBase,
                         public virtual yask_string_output {
    protected:
        ostringstream _oss;
    public:
        StringOutput() : OutputBase(_oss) {}
        
        virtual ~StringOutput() {
            discard();
        }

        string get_string() const {
            return _oss.str();
        }
        
        void discard() {
            _oss.str("");
        }
    };

    class StdoutOutput : public OutputBase,
                         public virtual yask_stdout_output {
    public:
        StdoutOutput() : OutputBase(cout) {}
    };

    class NullOutput : public OutputBase,
                       public virtual yask_null_output {
    protected:
        ofstream _ofs;          // never opened.
    public:
        NullOutput() : OutputBase(_ofs) {}
    };
    
    // See yask_common_api.hpp for documentation.
    yask_file_output_ptr
    yask_output_factory::new_file_output(const string& file_name) const {
        auto p = make_shared<FileOutput>();
        assert(p.get());
        p->open(file_name);
        return p;
    }
    yask_string_output_ptr
    yask_output_factory::new_string_output() const{
        auto p = make_shared<StringOutput>();
        assert(p.get());
        return p;
    }
    yask_stdout_output_ptr
    yask_output_factory::new_stdout_output() const{
        auto p = make_shared<StdoutOutput>();
        assert(p.get());
        return p;
    }
    yask_null_output_ptr
    yask_output_factory::new_null_output() const{
        auto p = make_shared<NullOutput>();
        assert(p.get());
        return p;
    }
    
}

