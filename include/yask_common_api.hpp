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

///////// APIs common to the YASK compiler and kernel. ////////////

// This file uses Doxygen 1.8 markup for API documentation-generation.
// See http://www.stack.nl/~dimitri/doxygen.
/** @file yask_common_api.hpp */ 

#ifndef YASK_COMMON_API
#define YASK_COMMON_API

#include <string>
#include <iostream>
#include <ostream>
#include <memory>

namespace yask {

    /**
     * \defgroup yask YASK Commmon Utilities
     * Types, clases, and functions used in both the \ref sec_yc and \ref sec_yk.
     * @{
     */

    /// Version information.
    /**
       @returns String describing the current version.
    */
    std::string yask_get_version_string();
    
    /// Type to use for indexing grids.
    /** Index types are signed to allow negative indices in padding/halos. */
#ifdef SWIG
    typedef long int idx_t;     // SWIG doesn't seem to understand int64_t.
#else
    typedef std::int64_t idx_t;
#endif

    // Forward declarations of class-pointers.

    class yask_output;
    /// Shared pointer to \ref yask_output
    typedef std::shared_ptr<yask_output> yask_output_ptr;

    class yask_file_output;
    /// Shared pointer to \ref yask_file_output
    typedef std::shared_ptr<yask_file_output> yask_file_output_ptr;

    class yask_string_output;
    /// Shared pointer to \ref yask_string_output
    typedef std::shared_ptr<yask_string_output> yask_string_output_ptr;

    class yask_stdout_output;
    /// Shared pointer to \ref yask_stdout_output
    typedef std::shared_ptr<yask_stdout_output> yask_stdout_output_ptr;

    class yask_null_output;
    /// Shared pointer to \ref yask_null_output
    typedef std::shared_ptr<yask_null_output> yask_null_output_ptr;

    /// Exception from yask framework
    /** Objects of this exception contain additional message from yask framework */
    class yask_exception: public std::exception {
    private:
    	/// Additional message container
    	std::string msg;

    public:
    	virtual ~yask_exception() {};

        /// Get default message.
        /** Returns a C-style character string describing the general cause of the current error.
            @returns default message of the exception. */
    	virtual const char* what() noexcept;

    	/// Add additional message to this exception.
    	void add_message(std::string message /**< [in] Additional message as string. */ );

        /// Get additional message.
        /** @returns additional message as string */
    	const char* get_message();
    };

    /// Factory to create output objects.
    class yask_output_factory {
    public:
        virtual ~yask_output_factory() {}

        /// Create a file output object.
        /**
           This object is used to write output to a file.
           @returns Pointer to new output object or null pointer if 
           file cannot be opened.
        */
        virtual yask_file_output_ptr
        new_file_output(const std::string& file_name
                        /**< [in] Name of file to open.
                           Any existing file will be truncated. */ ) const;

        /// Create a string output object.
        /**
           This object is used to write output to a string.
           @returns Pointer to new output object.
        */
        virtual yask_string_output_ptr
        new_string_output() const;

        /// Create a stdout output object.
        /**
           This object is used to write output to the standard output stream.
           @returns Pointer to new output object.
        */
        virtual yask_stdout_output_ptr
        new_stdout_output() const;

        /// Create a null output object.
        /**
           This object is used to discard output.
           @returns Pointer to new output object.
        */
        virtual yask_null_output_ptr
        new_null_output() const;
    };

    /// Base interface for output.
    class yask_output {
    public:
        virtual ~yask_output() {}

        /// Access underlying C++ ostream object.
        /** @returns Reference to ostream. */
        virtual std::ostream& get_ostream() =0;
    };

    /// File output.
    class yask_file_output : public virtual yask_output {
    public:
        virtual ~yask_file_output() {}

        /// Get the filename.
        /** @returns String containing filename given during creation. */
        virtual std::string get_filename() const =0;
        
        /// Close file.
        virtual void close() =0;
    };

    /// String output.
    class yask_string_output : public virtual yask_output {
    public:
        virtual ~yask_string_output() {}

        /// Get the output.
        /** Does not modify current buffer.
            @returns copy of current buffer's contents. */
        virtual std::string get_string() const =0;

        /// Discard contents of current buffer.
        virtual void discard() =0;
    };

    /// Stdout output.
    class yask_stdout_output : public virtual yask_output {
    public:
        virtual ~yask_stdout_output() {}
    };

    /// Null output.
    /** This object will discard all output. */
    class yask_null_output : public virtual yask_output {
    public:
        virtual ~yask_null_output() {}
    };

    /** @}*/

} // namespace yask.

#endif
