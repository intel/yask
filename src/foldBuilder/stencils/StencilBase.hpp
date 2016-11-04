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

// Base class for defining stencil equations.

#ifndef STENCIL_BASE
#define STENCIL_BASE

#include <map>
using namespace std;

class StencilBase;
typedef map<string, StencilBase*> StencilList;

class StencilBase {
protected:

    // Simple name for the stencil.
    string _name;
    
    // A grid is an n-dimensional space that is indexed by grid indices.
    // Vectorization will be applied to grid accesses.
    Grids _grids;       // keep track of all registered grids.

    // A parameter is an n-dimensional space that is NOT index by grid indices.
    // It is used to pass some sort of index-invarant setting to a stencil function.
    // Its indices must be resolved when define() is called.
    // At this time, this is not checked, so be careful!!
    Params _params;     // keep track of all registered non-grid vars.

public:
    virtual ~StencilBase() {}

    // Initialize name and register this new object in a list.
    StencilBase(const string name, StencilList& stencils) :
        _name(name)
    {
        stencils[_name] = this;
    }

    // Identification.
    virtual const string& getName() const { return _name; }
    
    // Get the registered grids and params.
    virtual Grids& getGrids() { return _grids; }
    virtual Grids& getParams() { return _params; }

    // Radius stub methods.
    virtual bool usesRadius() const { return false; }
    virtual bool setRadius(int radius) { return false; }
    virtual int getRadius() const { return 0; }

    // Define grid values relative to given offsets in each dimension.
    virtual void define(const IntTuple& offsets) = 0;
};

// A base class for stencils that have an 'radius'.
class StencilRadiusBase : public StencilBase {
protected:
    int _radius;         // stencil radius (for convenience; optional).

public:
    StencilRadiusBase(const string name, StencilList& stencils, int radius) :
        StencilBase(name, stencils), _radius(radius) {}

    // Does use radius.
    virtual bool usesRadius() const { return true; }
    
    // Set radius.
    // Return true if successful.
    virtual bool setRadius(int radius) {
        _radius = radius;
        return radius >= 0;  // support only non-neg. radius.
    }

    // Get radius.
    virtual int getRadius() { return _radius; }
};

#endif
