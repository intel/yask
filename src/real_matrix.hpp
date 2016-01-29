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

// This file defines a specific implementation of a RealMatrix.

// Define the following macros before including:
// - RMAT_CLASS: name of class.
// - RMAT_MAP: 4D->1D mapping.
// TODO: use more advanced C++ techniques to avoid macros.

// silly stringification trick.
#define RMAT_XSTR(s) RMAT_STR(s)
#define RMAT_STR(s) #s

template <long _numMats>
class RMAT_CLASS : public RealMatrixBase<_numMats> {

 public:
 
    RMAT_CLASS(long vlen_x, long vlen_y, long vlen_z,
             long d1, long d2, long d3,
             long p1, long p2, long p3,
             const string& name) :
    RealMatrixBase<_numMats>(vlen_x, vlen_y, vlen_z,
                              d1, d2, d3,
                              p1, p2, p3, name)
    {
        cout << " with '" << RMAT_XSTR(RMAT_MAP) << "' 4D->1D mapping scheme." << endl;
    }

    // 'line' args are for debug.
    // 'xv' args are vector indices.
    // 'x' args are individual REAL indices.
    // 'safe' should be set to true when pointer must be within matrix.
    // 'safe' should be set to false when pointer can be outside of matrix.

public:
    // Get pointer to a vector.
    // x,y,z are vector indices.
    ALWAYS_INLINE const realv* getPtr(int matNum, long xv, long yv, long zv, bool safe=true) const {
        long index = RMAT_MAP(xv + this->_p1v, yv + this->_p2v, zv + this->_p3v, matNum,
                            this->_s1v, this->_s2v, this->_s3v, _numMats);
#ifdef TRACE_MEM
        printf("getPtr(%d, %ld, %ld, %ld, %s) => %li\n",
               matNum, xv, yv, zv, safe ? "safe": "unsafe", index);
#endif
        if (safe) {
            assert(matNum >= 0); assert(matNum < _numMats);
            assert(xv >= -this->_p1v); assert(xv < this->_d1v + this->_p1v);
            assert(yv >= -this->_p2v); assert(yv < this->_d2v + this->_p2v);
            assert(zv >= -this->_p3v); assert(zv < this->_d3v + this->_p3v);
            assert(index >= 0); assert(index < this->_sizev);
        }
        return &this->_base[index];
    }

    // non-const version.
    ALWAYS_INLINE realv* getPtr(int matNum, long xv, long yv, long zv, bool safe=true) {
        const realv* p = const_cast<const RMAT_CLASS*>(this)->getPtr(matNum, xv, yv, zv, safe);
        return const_cast<realv*>(p);
    }

    // Read a vector.
    // x,y,z are vector indices.
    ALWAYS_INLINE realv readv(int matNum, long xv, long yv, long zv, int line=0) const {
        const realv* p = getPtr(matNum, xv, yv, zv);
        __assume_aligned(p, CACHELINE_BYTES);
        realv v;
        v.loadFrom(p);
#ifdef TRACE_MEM
        long x1 = xv * this->_vlen_x;
        long y1 = yv * this->_vlen_y;
        long z1 = zv * this->_vlen_z;
        printf("read vector %s[%i][%li..%li,%li..%li,%li..%li](%p) = [",
               this->_name.c_str(), matNum,
               x1,x1+this->_vlen_x-1, y1,y1+this->_vlen_y-1, z1,z1+this->_vlen_z-1, p);
        for (int i = 0; i < this->_vlen; i++) {
            if (i) printf(", ");
            printf("%.4f", v[i]);
        }
        printf("] at line %i.\n", line);
        fflush(stdout);
#endif
#ifdef MODEL_CACHE
        cache.read(p, line);
#endif
        return v;
    }

    // Write a vector.
    // x,y,z are vector indices.
    ALWAYS_INLINE void writev(const realv& v, int matNum, long xv, long yv, long zv, int line=0) {
        realv* p = getPtr(matNum, xv, yv, zv);
        __assume_aligned(p, CACHELINE_BYTES);
        v.storeTo(p);
#ifdef TRACE_MEM
        long x1 = xv*this->_vlen_x;
        long y1 = yv*this->_vlen_y;
        long z1 = zv*this->_vlen_z;
        printf("write vector %s[%i][%li..%li,%li..%li,%li..%li](%p) = [",
               this->_name.c_str(), matNum,
               x1,x1+this->_vlen_x-1,
               y1,y1+this->_vlen_y-1,
               z1,z1+this->_vlen_z-1, p);
        for (int i = 0; i < this->_vlen; i++) {
            if (i) printf(", ");
            printf("%.4f", v[i]);
        }
        printf("] at line %i.\n", line);
        fflush(stdout);
#endif
#ifdef MODEL_CACHE
        cache.write(p, line);
#endif
    }

    // single-element read--very slow.
    // x,y,z are element indices.
    REAL read1(int matNum, long x, long y, long z, int line=0) const {
        long xv, xo, yv, yo, zv, zo;
        FIX_INDEX_OFFSET(0, x, xv, xo, this->_vlen_x);
        FIX_INDEX_OFFSET(0, y, yv, yo, this->_vlen_y);
        FIX_INDEX_OFFSET(0, z, zv, zo, this->_vlen_z);
        const realv* p = getPtr(matNum, xv, yv, zv);
#ifdef TRACE_MEM
        printf("read element %s[%i][%li,%li,%li](%p) = %.4f at line %i.\n",
               this->_name.c_str(), matNum, x, y, z, p, (*p)(xo, yo, zo), line);
        fflush(stdout);
#endif
#ifdef MODEL_CACHE
        cache.read(p, line);
#endif
        return (*p)(xo, yo, zo);
    }

    // single-element write--very slow.
    // x,y,z are element indices.
    void write1(REAL val, int matNum, long x, long y, long z, int line=0) {
        long xv, xo, yv, yo, zv, zo;
        FIX_INDEX_OFFSET(0, x, xv, xo, this->_vlen_x);
        FIX_INDEX_OFFSET(0, y, yv, yo, this->_vlen_y);
        FIX_INDEX_OFFSET(0, z, zv, zo, this->_vlen_z);
        realv* p = getPtr(matNum, xv, yv, zv);
#ifdef TRACE_MEM
        printf("write element %s[%i][%li,%li,%li](%p) = %.4f at line %i.\n",
               this->_name.c_str(), matNum, x, y, z, p, val, line);
        fflush(stdout);
#endif
#ifdef MODEL_CACHE
        cache.write(p, line);
#endif
        (*p)(xo, yo, zo) = val;
    }

};

#undef RMAT_CLASS
#undef RMAT_MAP
