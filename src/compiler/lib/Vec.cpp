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

//////////// Basic vector classes /////////////

#include "Vec.hpp"

namespace yask {

    // Determine vectorizability information about this var point.
    // Called when a var point is read in a stencil function.
    string VecInfoVisitor::visit(VarPoint* gp) {

        // Nothing to do if this var-point is not vectorizable.
        if (gp->get_vec_type() == VarPoint::VEC_NONE) {
            #ifdef DEBUG_VV
            cout << " //** cannot vectorize scalar-access " << gp->make_quoted_str() << endl;
            #endif
            _scalar_points.insert(*gp);
            return "";
        }
        else if (gp->get_vec_type() == VarPoint::VEC_PARTIAL) {
            #ifdef DEBUG_VV
            cout << " //** cannot vectorize non-standard-access " << gp->make_quoted_str() << endl;
            #endif
            _non_vec_points.insert(*gp);
            return "";
        }
        assert(gp->get_vec_type() == VarPoint::VEC_FULL);

        // Already seen this point?
        if (_vec_points.count(*gp) > 0) {
            assert(_vblk2elem_lists.count(*gp) > 0);
            assert(_vblk2avblks.count(*gp) > 0);
            return "";
        }
        assert(_vblk2elem_lists.count(*gp) == 0);
        assert(_vblk2avblks.count(*gp) == 0);

        // Vec of points to calculate.
        #ifdef DEBUG_VV
        cout << " //** vec @ " << gp->make_quoted_str() << " => " << endl;
        #endif

        // Loop through all points in the vector fold.
        _dims._fold.visit_all_points
            ([&](const IntTuple& vec_point, size_t pelem) {

                 // Final offset in each dim is offset of var point plus
                 // fold offset.
                 // This works because we know this var point is accessed
                 // only by simple offsets in each foldable dim.
                 // Note: there may be more or fewer dims in vec_point than in var point.
                 auto offsets = gp->get_arg_offsets().add_elements(vec_point, false);

                 // Find aligned vector indices and offsets
                 // for this one point.
                 IntTuple vec_offsets, vec_location;
                 for (auto& dim : offsets) {
                     auto& dname = dim._get_name();

                     // length of this dimension in fold, if it exists.
                     const int* p = _dims._fold.lookup(dname);
                     int len = p ? *p : 1;

                     // convert this offset to vector index and vector offset.
                     int vec_index, vec_offset;
                     fix_index_offset(0, dim.get_val(), vec_index, vec_offset, len);
                     vec_offsets.add_dim_back(dname, vec_offset);
                     vec_location.add_dim_back(dname, vec_index * len);
                 }
                 #ifdef DEBUG_VV
                 cout << "  //** element @ " << offsets.make_dim_val_str() << " => " <<
                     " vec-location @ " << vec_location.make_dim_val_str() <<
                     " & vec-offsets @ " << vec_offsets.make_dim_val_str() <<
                     " => " << endl;
                 #endif

                 // Create aligned vector block that contains this point.
                 VarPoint aligned_vec = *gp;  // copy original.
                 aligned_vec.set_arg_offsets(vec_location);

                 // Find linear offset within this aligned vector block.
                 int aligned_elem = _dims._fold.layout(vec_offsets, false);
                 assert(aligned_elem >= 0);
                 assert(aligned_elem < _vlen);
                 #ifdef DEBUG_VV
                 cout << "   //** " << gp->make_str() << "[" << pelem << "] = aligned-" <<
                     aligned_vec.make_str() << "[" << aligned_elem << "]" << endl;
                 #endif

                 // Update set of *all* aligned vec-blocks.
                 _aligned_vecs.insert(aligned_vec);

                 // Update set of aligned vec-blocks and elements needed for *this* vec-block element.
                 _vblk2avblks[*gp].insert(aligned_vec);

                 // Save which aligned vec-block's element is needed for this vec-block element.
                 VecElem ve(aligned_vec, aligned_elem, offsets);
                 _vblk2elem_lists[*gp].push_back(ve); // should be at pelem index.
                 assert(_vblk2elem_lists[*gp].size() == pelem+1); // verify at pelem index.

                 return true;
             });                  // end of vector lambda-function.

        // Mark as done.
        _vec_points.insert(*gp);
        return "";
    }                   // end of visit() method.

    // Sort a commutative expression.
    string ExprReorderVisitor::visit(CommutativeExpr* ce) {

        auto& oev = ce->get_ops(); // old exprs.
        num_expr_ptr_vec nev; // new exprs.

        // Simple, greedy algorithm:
        // Select first element that needs the fewest new aligned vecs.
        // Repeat until done.
        // TODO: make more efficient--this is O(n^2), so it blows up
        // when there are long exprs with many reads.
        // TODO: sort based on all reused exprs, not just var reads.

        VarPointSet aligned_vecs; // aligned vecs needed so far.
        set<size_t> used_exprs; // expressions used.
        for (size_t i = 0; i < oev.size(); i++) {

            #ifdef DEBUG_SORT
            cout << "  Looking for expr #" << i << "..." << endl;
            #endif

            // Scan unused exprs.
            size_t j_best = 0;
            size_t j_best_cost = size_t(-1);
            VarPointSet j_best_aligned_vecs;
            for (size_t j = 0; j < oev.size(); j++) {
                if (used_exprs.count(j) == 0) {

                    // This unused expr.
                    auto& expr = oev[j];

                    // Get aligned vecs needed for this expr.
                    VecInfoVisitor tmpvv(_vv);
                    expr->accept(&tmpvv);
                    auto& tmp_aligned_vecs = tmpvv._aligned_vecs;

                    // Calculate cost.
                    size_t cost = 0;
                    for (auto k = tmp_aligned_vecs.begin(); k != tmp_aligned_vecs.end(); k++) {
                        auto& av = *k;

                        // new vector needed?
                        if (aligned_vecs.count(av) == 0) {
                            #ifdef DEBUG_SORT
                            cout << " Vec " << av.make_str("tmp") << " is new" << endl;
                            #endif
                            cost++;
                        }
                    }
                    #ifdef DEBUG_SORT
                    cout << " Cost of expr " << j << " = " << cost << endl;
                    #endif
                    // Best so far?
                    if (cost < j_best_cost) {
                        j_best_cost = cost;
                        j_best = j;
                        j_best_aligned_vecs = tmp_aligned_vecs;
                        #ifdef DEBUG_SORT
                        cout << "  Best so far has " << j_best_aligned_vecs.size() << " aligned vecs" << endl;
                        #endif
                    }
                }
            }

            // Must have a best one.
            assert(j_best_cost != size_t(-1));

            // Add it.
            nev.push_back(oev[j_best]);
            used_exprs.insert(j_best);

            // Remember used vectors.
            for (auto k = j_best_aligned_vecs.begin(); k != j_best_aligned_vecs.end(); k++) {
                aligned_vecs.insert(*k);
            }
        }

        // Replace the old vector w/the new one.
        assert(nev.size() == oev.size());
        oev.swap(nev);
        return "";
    }

    // TODO: fix this old code and make it available as an output.
    #if 0
    // Print stats for various folding options.
    if (vlen_for_stats) {
        string separator(",");
        VecInfoVisitor::print_stats_header(cout, separator);

        // Loop through all vars.
        for (auto gp : vars) {

            // Loop through possible folds of given length.
            for (int xlen = vlen_for_stats; xlen > 0; xlen--) {
                for (int ylen = vlen_for_stats / xlen; ylen > 0; ylen--) {
                    int zlen = vlen_for_stats / xlen / ylen;
                    if (vlen_for_stats == xlen * ylen * zlen) {

                        // Create vectors needed to implement RHS.
                        VecInfoVisitor vv(xlen, ylen, zlen);
                        gp->visit_exprs(&vv);

                        // Print stats.
                        vv.print_stats(cout, gp->_get_name(), separator);
                    }
                }
            }
        }
    }
    #endif

} // namespace yask.
