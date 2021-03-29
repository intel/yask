/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2021, Intel Corporation

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

//////////// Basic vector classes /////////////

#pragma once

#include "Print.hpp"

namespace yask {

    //#define DEBUG_VV   // debug VecInfoVisitor.

    // One element in a vector.
    class VecElem {
    public:
        VarPoint _vec;      // starting index of vector containing this element.
        size_t _offset;      // 1-D offset in _vec.
        IntTuple _offsets;   // n-D offsets.

        VecElem(const VarPoint& vec, int offset, const IntTuple& offsets) :
            _vec(vec), _offset(offset), _offsets(offsets) { }
        virtual ~VecElem() { }

        virtual bool operator==(const VecElem& rhs) const {
            return _vec == rhs._vec && _offset == rhs._offset;
        }
        virtual bool operator!=(const VecElem& rhs) const {
            return !operator==(rhs);
        }
    };

    // STL vector of vector block elements.
    // The STL vector index is the HW vector index.
    // Each entry indicates in what aligned vector and offset
    // the element can be found.
    typedef vector<VecElem> VecElemList;

    // Map of each vector block to aligned block(s) that contain its points.
    typedef map<VarPoint, VarPointSet> Point2Vecs;

    // Map of each vector block to its elements.
    typedef map<VarPoint, VecElemList> Point2VecElemLists;

    // This visitor determines the vector blocks needed to calculate the stencil.
    // It doesn't actually generate code; it just collects info from the AST.
    // After the AST has been visited, the object can be used to print statistics
    // or used to provide data to a print visitor.
    class VecInfoVisitor : public ExprVisitor {
    protected:

        const Dimensions& _dims;
        int _vlen;                   // size of one vector.

    public:

        // Data on vectorizable points.
        VarPointSet _aligned_vecs; // set of aligned vectors, i.e., ones that need to be read from memory.
        Point2Vecs _vblk2avblks; // each vec block -> its constituent aligned vec blocks.
        Point2VecElemLists _vblk2elem_lists; // each vec block -> in-order list of its aligned vec blocks' elements.
        VarPointSet _vec_points;            // set of all vectorizable points read from, aligned and unaligned.
        VarPointSet _vec_writes;            // set of vectors written to.

        // NB: the above hold much of the same info, but arranged differently:
        // _aligned_vecs contains only a set of aligned blocks.
        // _vblk2avblks is used to quickly determine which aligned blocks contribute to a given block.
        // _vblk2elem_lists is used to find exactly where each element comes from.
        // The keys are the same for both maps: the vectorizable var points.

        // Data on non-vectorizable points.
        VarPointSet _scalar_points; // set of points that should be read as scalars and broadcast to vectors.
        VarPointSet _non_vec_points; // set of points that are not scalars or vectorizable.

        VecInfoVisitor(const Dimensions& dims) :
            _dims(dims) {
            _vlen = dims._fold.product();
        }

        virtual const IntTuple& get_fold() const {
            return _dims._fold;
        }

        virtual size_t get_num_points() const {
            return _vblk2elem_lists.size();
        }

        virtual size_t get_num_aligned_vecs() const {
            return _aligned_vecs.size();
        }

        // Make an index and offset canonical, i.e.,
        // find index_out and offset_out such that
        // (index_out * vec_len) + offset_out == (index_in * vec_len) + offset_in
        // and offset_out is in [0..vec_len-1].
        // Makes proper adjustments for negative inputs.
        virtual void fix_index_offset(int index_in, int offset_in,
                                    int& index_out, int& offset_out,
                                    int vec_len) {
            const int ofs = (index_in * vec_len) + offset_in;
            index_out = ofs / vec_len;
            offset_out = ofs % vec_len;
            while(offset_out < 0) {
                offset_out += vec_len;
                index_out--;
            }
        }

#if 0
        // Print stats header.
        virtual void print_stats_header(ostream& os, string separator) const {
            os << "destination var" <<
                separator << "num vector elems" <<
                separator << "fold" <<
                separator << "num points in stencil" <<
                separator << "num aligned vectors to read from memory" <<
                separator << "num blends needed" <<
                separator << _dims._fold.make_dim_str(separator, "footprint in ") <<
                endl;
        }

        // Print some stats for the current values.
        // Pre-requisite: visitor has been accepted.
        virtual void print_stats(ostream& os, const string& dest_var, const string& separator) const {

            // calc num blends needed.
            int num_blends = 0;
            for (auto i = _vblk2elem_lists.begin(); i != _vblk2elem_lists.end(); i++) {
                //auto pt = i->first;
                auto vev = i->second;
                VarPointSet mems; // inputs for this point.
                for (size_t j = 0; j < vev.size(); j++)
                    mems.insert(vev[j]._vec);
                if (mems.size() > 1) // only need to blend if >1 inputs.
                    num_blends += mems.size();
            }

            // calc footprint in each dim.
            map<const string, int> footprints;
            for (auto& dim : _dims._fold) {
                auto& dname = dim._get_name();

                // Make set of aligned vecs projected in this dir.
                set<IntTuple> footprint;
                for (auto vb : _aligned_vecs) {
                    IntTuple proj = vb.remove_dim(dim);
                    footprint.insert(proj);
                }
                footprints[dname] = footprint.size();
            }

            os << dest_var <<
                separator << _vlen <<
                separator << _dims._fold.make_val_str("*") <<
                separator << get_num_points() <<
                separator << get_num_aligned_vecs() <<
                separator << num_blends;
            for (auto& dim : _dims._fold)
                os << separator << footprints[dim._get_name()];
            os << endl;
        }
#endif

        // Equality.
        virtual string visit(EqualsExpr* ee) {

            // Only want to continue visit on RHS of an eq_group.
            ee->_get_rhs()->accept(this);

            // For LHS, just save point.
            auto lhs = ee->_get_lhs();
            _vec_writes.insert(*lhs);
            return "";
        }

        // Called when a var point is read in a stencil function.
        virtual string visit(VarPoint* gp);
    };

    // Define methods for printing a vectorized version of the stencil.
    class VecPrintHelper {
    protected:
        VecInfoVisitor& _vv;
        bool _reuse_vars; // if true, load to a local var; else, reload on every access.
        map<VarPoint, string> _vec_vars; // vecs that are already constructed (key is var point).
        map<string, string> _elem_vars; // elems that are already read (key is read stmt).

        // Print access to an aligned vector block.
        // Return var name.
        virtual string print_aligned_vec_read(ostream& os, const VarPoint& gp) =0;

        // Print unaligned vector memory read.
        // Assumes this results in same values as print_unaligned_vec().
        // Return var name.
        virtual string print_unaligned_vec_read(ostream& os, const VarPoint& gp) =0;

        // Print write to an aligned vector block.
        virtual void print_aligned_vec_write(ostream& os, const VarPoint& gp,
                                             const string& val) =0;

        // Print conversion from existing vars to make an unaligned vector block.
        // Return var name.
        virtual string print_unaligned_vec(ostream& os, const VarPoint& gp) =0;

        // Print construction for one point var pv_name from elems.
        virtual void print_unaligned_vec_ctor(ostream& os, const VarPoint& gp,
                                              const string& pv_name) =0;

        // Read from a single point.
        // Return code for read.
        virtual string read_from_scalar_point(ostream& os, const VarPoint& gp,
                                              const VarMap& v_map) =0;

        // Read from multiple points that are not vectorizable.
        // Return var name.
        virtual string print_partial_vec_read(ostream& os, const VarPoint& gp) =0;

    public:
        VecPrintHelper(VecInfoVisitor& vv,
                       bool reuse_vars = true) :
            _vv(vv), _reuse_vars(reuse_vars) { }
        virtual ~VecPrintHelper() {}

        // get fold info.
        virtual const IntTuple& get_fold() const {
            return _vv.get_fold();
        }

        // Return point info.
        virtual bool is_aligned(const VarPoint& gp) {
            return _vv._aligned_vecs.count(gp) > 0;
        }

        // Access cached values.
        virtual const string* save_point_var(const VarPoint& gp, const string& var) {
            _vec_vars[gp] = var;
            return &_vec_vars.at(gp);
        }
        virtual const string* lookup_point_var(const VarPoint& gp) {
            if (_vec_vars.count(gp))
                return &_vec_vars.at(gp);
            return 0;
        }
        virtual const string* save_elem_var(const string& expr, const string& var) {
            _elem_vars[expr] = var;
            return &_elem_vars.at(expr);
        }
        virtual const string* lookup_elem_var(const string& expr) {
            if (_elem_vars.count(expr))
                return &_elem_vars.at(expr);
            return 0;
        }
    };

    // A visitor that reorders exprs based on vector info.
    class ExprReorderVisitor : public ExprVisitor {
    protected:
        VecInfoVisitor& _vv;

    public:
        ExprReorderVisitor(VecInfoVisitor& vv) :
            _vv(vv) { }
        virtual ~ExprReorderVisitor() {}

        // Sort a commutative expression.
        virtual string visit(CommutativeExpr* ce);
    };

} // namespace yask.

