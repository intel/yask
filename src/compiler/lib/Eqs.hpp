/*****************************************************************************

YASK: Yet Another Stencil Kit
Copyright (c) 2014-2022, Intel Corporation

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

///////// Classes for equations, equation bundles, and stages. ////////////

#pragma once

#include "Expr.hpp"
#include "VarPoint.hpp"
#include "Settings.hpp"

using namespace std;

namespace yask {

    // Dependencies between objects of type T.
    template <typename T>
    class Deps {

    public:
        typedef shared_ptr<T> Tp;
        typedef unordered_set<Tp> TpSet;
        typedef vector_set<Tp> TpList;

        // dep_map[A].count(B) > 0 => A depends on B.
        typedef unordered_map<Tp, TpSet> DepMap;

    protected:
        DepMap _imm_deps;       // immediate deps, i.e., transitive reduction.
        DepMap _full_deps;      // transitive closure of _imm_deps.
        TpSet _all;             // set of all objs.
        bool _done = false;     // transitive closure done?
        TpSet _empty;

        // Recursive helper for visit_deps().
        virtual void _visit_deps(Tp a,
                                std::function<void (Tp b, TpList& path)> visitor,
                                TpList* seen) const {

            // Already visited, i.e., a loop?
            bool was_seen = (seen && seen->count(a));
            if (was_seen)
                return;

            // Add 'a' to copy of path.
            // Important to make a separate copy for recursive calls.
            TpList seen1;
            if (seen)
                seen1 = *seen; // copy nodes already seen.
            seen1.insert(a);   // add this one.

            // Call lambda fn.
            visitor(a, seen1);

            // any dependencies?
            if (_imm_deps.count(a)) {
                auto& adeps = _imm_deps.at(a);

                // loop thru deps of 'a', i.e., each 'b' deps on 'a'.
                for (auto b : adeps) {

                    // Recurse to deps of 'b'.
                    _visit_deps(b, visitor, &seen1);
                }
            }
        }

    public:

        Deps() {}
        virtual ~Deps() {}

        // Declare that eq a depends directly on b.
        virtual void set_imm_dep_on(Tp a, Tp b) {
            _imm_deps[a].insert(b);
            _all.insert(a);
            _all.insert(b);
            _done = false;
        }

        // Clear all deps.
        virtual void clear_deps() {
            _imm_deps.clear();
            _full_deps.clear();
            _all.clear();
            _done = true;
        }

        // Check whether eq a directly depends on b.
        virtual bool is_imm_dep_on(Tp a, Tp b) const {
            return _imm_deps.count(a) && _imm_deps.at(a).count(b) > 0;
        }

        // Checks for immediate dependencies in either direction.
        virtual bool is_imm_dep(Tp a, Tp b) const {
            return is_imm_dep_on(a, b) || is_imm_dep_on(b, a);
        }

        // Check whether eq 'a' depends on 'b'.
        virtual bool is_dep_on(Tp a, Tp b) const {
            assert(_done);
            return _full_deps.count(a) && _full_deps.at(a).count(b) > 0;
        }

        // Checks for dependencies in either direction.
        virtual bool is_dep(Tp a, Tp b) const {
            return is_dep_on(a, b) || is_dep_on(b, a);
        }

        // Get all the objects that 'a' depends on.
        virtual const TpSet& get_imm_deps_on(Tp a) const {
            if (_imm_deps.count(a) == 0)
                return _empty;
            return _imm_deps.at(a);
        }
        virtual const TpSet& get_deps_on(Tp a) const {
            assert(_done);
            if (_full_deps.count(a) == 0)
                return _empty;
            return _full_deps.at(a);
        }

        // Visit 'a' and all its dependencies.
        // At each dep node 'b' in graph, 'visitor(b, path)' is called,
        // where 'path' contains all nodes from 'a' thru 'b' in dep order.
        virtual void visit_deps(Tp a,
                               std::function<void (Tp b,
                                                   TpList& path)> visitor) const {
            _visit_deps(a, visitor, NULL);
        }

        // Print deps. 'T' must implement get_descr().
        virtual void print_deps(ostream& os) const {
            os << "Dependencies within " << _all.size() << " objects:\n";
            for (auto& a : _all) {
                os << " For " << a->get_descr() << ":\n";
                visit_deps(a, [&](Tp b, TpList& path) {
                        if (a == b)
                            os << "  depends on self";
                        else
                            os << "  depends on " << b->get_descr();
                        os << " w/path of length " << path.size() << endl;
                    });
            }
        }

        // Does recursive analysis to find transitive closure.
        virtual void find_all_deps() {
            if (_done)
                return;
            for (auto a : _all)
                if (_full_deps.count(a) == 0)
                    visit_deps(a, [&](Tp b, TpList& path) {

                            // Walk path from ee to b.
                            // Every 'eq' in 'path' before 'b' depends on 'b'.
                            for (auto eq : path)
                                if (eq != b)
                                    _full_deps[eq].insert(b);
                        });
            _done = true;
        }
    };

    // A set of objects that have inter-dependencies.
    // Class 'T' must implement 'clone()' that returns
    // a 'shared_ptr<T>'.
    template <typename T>
    class DepGroup {

    public:
        typedef shared_ptr<T> Tp;
        typedef unordered_set<Tp> TpSet;
        typedef vector_set<Tp> TpList;

    protected:

        // Objs in this group.
        TpList _all;

        // Dependencies between all objs.
        Deps<T> _deps;

        // Dependencies from non-scratch objs to/on scratch objs.
        // NB: just using Deps::_imm_deps as a simple map of sets.
        Deps<T> _scratches;

    public:

        DepGroup() { }
        virtual ~DepGroup() { }

        // Assigning a DepGroup will clone its items
        // but drop all deps.
        // TODO: clone the deps, too.
        DepGroup& operator=(const DepGroup& src) {
            for (auto& p : src._all)
                _all.insert(p->clone());
            _deps.clear_deps();
            _scratches.clear_deps();
            return *this;
        }

        // list accessors.
        virtual void add_item(Tp p) {
            _all.insert(p);
        }
        virtual const TpList& get_all() const {
            return _all;
        }
        virtual int get_num() const {
            return _all.size();
        }

        // Get the deps.
        virtual const Deps<T>& get_deps() const {
            return _deps;
        }
        virtual Deps<T>& get_deps() {
            return _deps;
        }
        virtual const TpSet& get_deps(Tp p) const {
            return _deps.get_deps_on(p);
        }

        // Get the scratch deps.
        virtual const Deps<T>& get_scratch_deps() const {
            return _scratches;
        }
        virtual Deps<T>& get_scratch_deps() {
            return _scratches;
        }
        virtual const TpSet& get_scratch_deps(Tp p) const {
            return _scratches.get_deps_on(p);
        }

        // Find indirect dependencies based on direct deps.
        virtual void find_all_deps() {
            _deps.find_all_deps();
            _scratches.find_all_deps();
        }

        // Reorder based on dependencies,
        // i.e., topological sort.
        virtual void topo_sort() {
            find_all_deps();

            // No need to sort less than two things.
            if (_all.size() <= 1)
                return;

            // Want to keep original order as much as possible.
            // Only reorder if dependencies are in conflict.

            // Scan from beginning to next-to-last.
            for (size_t i = 0; i < _all.size() - 1; i++) {

                // Repeat until no dependent found.
                bool done = false;
                while (!done) {

                    // Does obj[i] depend on any obj after it?
                    auto& oi = _all.at(i);
                    done = true;
                    for (size_t j = i+1; j < _all.size(); j++) {
                        auto& oj = _all.at(j);

                        // Must swap if dependent.
                        if (_deps.is_dep_on(oi, oj)) {

                            // Error if also back-dep.
                            if (_deps.is_dep_on(oj, oi)) {
                                THROW_YASK_EXCEPTION("circular dependency between " +
                                                     oi->get_descr() + " and " +
                                                     oj->get_descr());
                            }

                            // Swap them.
                            _all.swap(i, j);

                            // Start over at index i.
                            done = false;
                            break;
                        }
                    }
                }
            }
        }

        // Copy dependencies from the 'full' graph to this
        // condensed graph.
        // Class 'T' must implement 'get_items()', which returns
        // an iteratable container of 'Tf' types.
        // See https://en.wikipedia.org/wiki/Directed_acyclic_graph.
        template <typename Tf>
        void inherit_deps_from(const DepGroup<Tf>& full) {

            // Deps between Tf objs.
            auto& fdeps = full.get_deps();
            auto& fscrs = full.get_scratch_deps();

            // All T objs in this.
            for (auto& oi : _all) {

                // All other T objs in this.
                for (auto& oj : _all) {

                    // Don't compare to self.
                    if (oi == oj) continue;

                    // All Tf objs in 'oi'.
                    for (auto& foi : oi->get_items()) {

                        // All Tf objs in 'oj'.
                        for (auto& foj : oj->get_items()) {

                            // If 'foi' is dep on 'foj',
                            // then 'oi' is dep on 'oj'.
                            if (fdeps.is_imm_dep_on(foi, foj))
                                _deps.set_imm_dep_on(oi, oj);
                            if (fscrs.is_imm_dep_on(foi, foj))
                                _scratches.set_imm_dep_on(oi, oj);
                        }
                    }
                }
            }
            find_all_deps();
        }
    };

    // A list of unique equation ptrs.
    typedef vector_set<equals_expr_ptr> EqList;

    // A set of equations and related dependency data.
    class Eqs : public DepGroup<EqualsExpr> {

    public:

        // Visit all equations.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& ep : _all) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.
        virtual void analyze_eqs(const CompilerSettings& settings,
                                 Dimensions& dims,
                                 std::ostream& os);

        // Determine which var points can be vectorized.
        virtual void analyze_vec(const CompilerSettings& settings,
                                 const Dimensions& dims);

        // Determine how var points are accessed in a loop.
        virtual void analyze_loop(const CompilerSettings& settings,
                                  const Dimensions& dims);

        // Update var access stats.
        virtual void update_var_stats();

        // Find scratch-var eqs needed for each non-scratch eq.
        virtual void analyze_scratch();
    };

    // A collection that holds various independent eqs.
    class EqLot {
    protected:
        EqList _eqs;            // all equations.
        Vars _out_vars;        // vars updated by _eqs.
        Vars _in_vars;         // vars read from by _eqs.
        bool _is_scratch = false; // true if _eqs update temp var(s).

    public:

        // Parts of the name.
        // TODO: move these into protected section and make accessors.
        string base_name;            // base name of this bundle.
        int index;                  // index to distinguish repeated names.

        // Ctor.
        EqLot(bool is_scratch) : _is_scratch(is_scratch) { }
        virtual ~EqLot() {}

        // Get all eqs.
        virtual const EqList& get_eqs() const {
            return _eqs;
        }

        // Visit all the equations.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
                ep->accept(ev);
            }
        }

        // Get the full name.
        virtual string _get_name() const;

        // Get number of equations.
        virtual int get_num_eqs() const {
            return _eqs.size();
        }

        // Updating temp vars?
        virtual bool is_scratch() const { return _is_scratch; }

        // Get vars output and input.
        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }
        virtual const Vars& get_input_vars() const {
            return _in_vars;
        }

        // Print stats for the equation(s).
        virtual void print_stats(ostream& os, const string& msg);
    };

    // A named equation bundle, which contains one or more var-update
    // equations.  All equations in a bundle must have the same conditions.
    // Equations in a bundle must not have inter-dependencies because they
    // will be combined into a single expression.
    class EqBundle : public EqLot {
    protected:
        const Dimensions* _dims = 0;

    public:

        // TODO: move these into protected section and make accessors.

        // Common conditions.
        bool_expr_ptr cond;
        bool_expr_ptr step_cond;

        // Common step expr.
        num_expr_ptr step_expr;

        // Create a copy containing clones of the equations.
        virtual shared_ptr<EqBundle> clone() const {
            auto p = make_shared<EqBundle>(*_dims, _is_scratch);

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->_eqs.clear();
            for (auto& i : _eqs)
                p->_eqs.insert(i->clone());

            return p;
        }

        // Ctor.
        EqBundle(const Dimensions& dims, bool is_scratch) :
            EqLot(is_scratch), _dims(&dims) { }
        virtual ~EqBundle() {}

        // Get a string description.
        virtual string get_descr(bool show_cond = true,
                                string quote = "'") const;

        // Add an equation to this bundle.
        virtual void add_eq(equals_expr_ptr ee);

        // Get the list of all equations.
        virtual const EqList& get_items() const {
            return _eqs;
        }

        // Visit the condition.
        // Return true if there was one to visit.
        virtual bool visit_cond(ExprVisitor* ev) {
            if (cond.get()) {
                cond->accept(ev);
                return true;
            }
            return false;
        }

        // Visit the step condition.
        // Return true if there was one to visit.
        virtual bool visit_step_cond(ExprVisitor* ev) {
            if (step_cond.get()) {
                step_cond->accept(ev);
                return true;
            }
            return false;
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicate_eqs_in_cluster(Dimensions& dims);
    };

    // Container for multiple equation bundles.
    class EqBundles : public DepGroup<EqBundle> {
    protected:

        // Copy of some global data.
        string _basename_default;
        Dimensions* _dims = 0;

        // Track vars that are updated.
        Vars _out_vars;

        // Map to track indices per eq-bundle name.
        map<string, int> _indices;

        // Track equations that have been added already.
        set<equals_expr_ptr> _eqs_in_bundles;

        // Add 'eq' from 'eqs' to eq-bundle with 'base_name'
        // unless already added or illegal.  The corresponding index in
        // '_indices' will be incremented if a new bundle is created.
        // Returns whether a new bundle was created.
        virtual bool add_eq_to_bundle(Eqs& eqs,
                                   equals_expr_ptr eq,
                                   const string& base_name,
                                   const CompilerSettings& settings);

    public:
        EqBundles() {}
        EqBundles(const string& basename_default, Dimensions& dims) :
            _basename_default(basename_default),
            _dims(&dims) {}
        virtual ~EqBundles() {}

        virtual void set_basename_default(const string& basename_default) {
            _basename_default = basename_default;
        }
        virtual void set_dims(Dimensions& dims) {
            _dims = &dims;
        }

        // Separate a set of equations into eq_bundles based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "eq_bundle1=foo,eq_bundle2=bar".
        // In this example, all eqs updating var names containing 'foo' go in eq_bundle1,
        // all eqs updating var names containing 'bar' go in eq_bundle2, and
        // each remaining eq goes into a separate eq_bundle.
        void make_eq_bundles(Eqs& eqs,
                           const CompilerSettings& settings,
                           std::ostream& os);

        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all eq_bundles.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& eg : _all)
                eg->visit_eqs(ev);
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicate_eqs_in_cluster(Dimensions& dims) {
            for (auto& eg : _all)
                eg->replicate_eqs_in_cluster(dims);
        }

        // Print stats for the equation(s) in all bundles.
        virtual void print_stats(ostream& os, const string& msg);

        // Apply optimizations requested in settings.
        virtual void optimize_eq_bundles(CompilerSettings& settings,
                                       const string& descr,
                                       bool print_sets,
                                       ostream& os);
    };

    typedef shared_ptr<EqBundle> EqBundlePtr;

    // A list of unique equation bundles.
    typedef vector_set<EqBundlePtr> EqBundleList;

    // A named equation stage, which contains one or more equation
    // bundles.  All equations in a stage do not need to have the same
    // domain condition, but they must have the same step condition.
    // Equations in a stage must not have inter-dependencies because they
    // may be run in parallel or in any order on any sub-domain.
    class EqStage : public EqLot {
    protected:
        EqBundleList _bundles;  // bundles in this stage.

    public:

        // Common condition.
        bool_expr_ptr step_cond;

        // Ctor.
        EqStage(bool is_scratch) :
            EqLot(is_scratch) { }
        virtual ~EqStage() { }

        // Create a copy containing clones of the bundles.
        virtual shared_ptr<EqStage> clone() const {
            auto p = make_shared<EqStage>(_is_scratch);

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->_eqs.clear();
            for (auto& i : _bundles)
                p->_bundles.insert(i->clone());

            return p;
        }

        // Get a string description.
        virtual string get_descr(string quote = "'") const;

        // Add a bundle to this stage.
        virtual void add_bundle(EqBundlePtr ee);

        // Get the list of all bundles
        virtual const EqBundleList& get_bundles() const {
            return _bundles;
        }
        virtual const EqBundleList& get_items() const {
            return _bundles;
        }

        // Visit the step condition.
        // Return true if there was one to visit.
        virtual bool visit_step_cond(ExprVisitor* ev) {
            if (step_cond.get()) {
                step_cond->accept(ev);
                return true;
            }
            return false;
        }

    };

    // Container for multiple equation stages.
    class EqStages : public DepGroup<EqStage> {
    protected:
        string _base_name = "stage";

        // Bundle index.
        int _idx = 0;

        // Track vars that are updated.
        Vars _out_vars;

        // Track bundles that have been added already.
        set<EqBundlePtr> _bundles_in_stages;

        // Add 'bp' from 'all_bundles'. Create new stage if needed.  Returns
        // whether a new stage was created.
        bool add_bundle_to_stage(EqBundles& all_bundles,
                             EqBundlePtr bp);

    public:

        // Separate bundles into stages.
        void make_stages(EqBundles& bundles,
                       std::ostream& os);

        // Get all output vars.
        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all stages.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& bp : _all)
                bp->visit_eqs(ev);
        }

        // Find halos needed for each var.
        virtual void calc_halos(EqBundles& all_bundles);

    }; // EqStages.

} // namespace yask.

