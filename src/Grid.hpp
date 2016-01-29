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

// Grid is made of reals.
typedef REAL GridValue;

// A 3D (x, y, z) collection of GridValues.
class Grid3d {
    
protected:
    VEL_MAT_TYPE _data;

public:

    // Ctor.
    Grid3d(long vlen_x, long vlen_y, long vlen_z,
         long d1, long d2, long d3,
         long p1, long p2, long p3,
         const string& name) :
        _data(vlen_x, vlen_y, vlen_z,
              d1, d2, d3,
              p1, p2, p3,
              name) { }

    // Access to data.
    const VEL_MAT_TYPE& getData() const {
        return _data;
    }
    VEL_MAT_TYPE& getData() {
        return _data;
    }

    // Read one element of the grid for testing.
    REAL readVal(long i, long j, long k, int line) const {
        int n = 0;
        return _data.read1(n, i, j, k, line);
    }

    // Implements read operator().
    GridValue operator()(long i, long j, long k) const {
        return readVal(i, j, k, __LINE__);
    }

    // Write one element of the grid for testing.
    void writeVal(REAL val, long i, long j, long k, int line=0) {
        int n = 0;
        _data.write1(val, n, i, j, k, line);
    }

    // Read one vector at xv, yv, zv.
    ALWAYS_INLINE const realv readVec(long xv, long yv, long zv, int line=0) const {
        int n = 0;
        return _data.readv(n, xv, yv, zv, line);
    }

    // Write one vector at t, xv, yv, zv.
    ALWAYS_INLINE void writeVec(const realv& v, long xv, long yv, long zv, int line=0) {
        int n = 0;
        _data.writev(v, n, xv, yv, zv, line);
    }

    // Get a pointer to vector at t, xv, yv, zv.
    ALWAYS_INLINE const realv* getVecPtr(long i, long j, long k, int line=0) const {
        int n = 0;
        return _data.getPtr(n, i, j, k, false);
    }
};

// A 5D (time, var, x, y, z) collection of GridValues.
// Currently only allows read from t and write to t+TIME_STEPS.
class Grid5d {
    
protected:
    MATRIX_TYPE _data;        // all the current and next values.

public:

    // Ctor.
    Grid5d(long vlen_x, long vlen_y, long vlen_z,
         long d1, long d2, long d3,
         long p1, long p2, long p3,
         const string& name) :
        _data(vlen_x, vlen_y, vlen_z,
              d1, d2, d3,
              p1, p2, p3,
              name) { }

    // Access to data.
    const MATRIX_TYPE& getData() const {
        return _data;
    }
    MATRIX_TYPE& getData() {
        return _data;
    }

    // Get correct index based on time & variable number.
    // There will be overlap between indices if NUM_WORKS < NUM_VARS.
    ALWAYS_INLINE int getMatIndex(int t, int var) const {
        assert(t % TIME_STEPS == 0);

        /* Example with NUM_VARS=5 & NUM_WORKS=2:
           t=0...
           var 0: mat 0 => mat 5
           var 1: mat 1 => mat 6
           var 2: mat 2 => mat 0
           var 3: mat 3 => mat 1
           var 4: mat 4 => mat 2
           t=1...
           var 0: mat 5 => mat 3
           var 1: mat 6 => mat 4
           var 2: mat 0 => mat 5
           var 3: mat 1 => mat 6
           var 4: mat 2 => mat 0
        */

        return ((t / TIME_STEPS) * NUM_VARS + var) % NUM_GRIDS;
    }

    // Read one element of the grid for testing.
    REAL readVal(int t, int var, long i, long j, long k, int line) const {
        int n = getMatIndex(t, var);
        return _data.read1(n, i, j, k, line);
    }

    // Implements read operator().
    GridValue operator()(int t, int var, long i, long j, long k) const {
        return readVal(t, var, i, j, k, __LINE__);
    }

    // Write one element of the grid for testing.
    void writeVal(REAL val, int t, int var, long i, long j, long k, int line=0) {
        int n = getMatIndex(t, var);
        _data.write1(val, n, i, j, k, line);
    }

    // Read one vector at xv, yv, zv.
    ALWAYS_INLINE const realv readVec(int t, int var, long xv, long yv, long zv, int line=0) const {
        int n = getMatIndex(t, var);
        return _data.readv(n, xv, yv, zv, line);
    }

    // Write one vector at t, xv, yv, zv.
    ALWAYS_INLINE void writeVec(const realv& v, int t, int var, long xv, long yv, long zv, int line=0) {
        int n = getMatIndex(t, var);
        _data.writev(v, n, xv, yv, zv, line);
    }

    // Get a pointer to vector at t, xv, yv, zv.
    ALWAYS_INLINE const realv* getVecPtr(int t, int var, long i, long j, long k, int line=0) const {
        int n = getMatIndex(t, var);
        return _data.getPtr(n, i, j, k, false);
    }
};
