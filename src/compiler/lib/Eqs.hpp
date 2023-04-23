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

///////// Classes for equations, equation bundles, and stages. ////////////

#pragma once

#include "Expr.hpp"
#include "VarPoint.hpp"
#include "Settings.hpp"

using namespace std;

namespace yask {

    using Eq = equals_expr_ptr;
    using EqList = vector_set<Eq>;

    template<typename T> using Tp = shared_ptr<T>;
    template<typename T> using TpSet = unordered_set<Tp<T>>;
    template<typename T> using TpList = vector_set<Tp<T>>;
    template<typename T> using TpSetMap = unordered_map<Tp<T>, TpSet<T>>;
    
    // Dependencies between objects of type T.
    template <typename T>
    class Deps {
        
    public:
        // dep_map[A].count(B) > 0 => A depends on B.
        typedef TpSetMap<T> DepMap;

    private:
        TpSet<T> _all;          // set of all objs.
        DepMap _imm_deps;       // immediate deps, i.e., transitive reduction.
        DepMap _full_deps;      // transitive closure of _imm_deps.
        bool _done = false;     // transitive closure done?
        TpSet<T> _empty;        // for returning a ref to an empty set.

        // Recursive helper for visit_deps().
        virtual void _visit_deps(Tp<T> a,
                                 std::function<void (Tp<T> b, TpList<T>& path)> visitor,
                                 TpList<T>* seen) const {

            // Already visited, i.e., a loop?
            bool was_seen = (seen && seen->count(a));
            if (was_seen)
                return;

            // Add 'a' to copy of path.
            // Important to make a separate copy for recursive calls.
            TpList<T> seen1;
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

        // Get deps.
        virtual const DepMap& get_imm_deps() const {
            return _imm_deps;
        }
        virtual const DepMap& get_all_deps() const {
            return _full_deps;
        }

        // Declare that eq a depends directly on b.
        virtual void set_imm_dep_on(Tp<T> a, Tp<T> b) {
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
        virtual bool is_imm_dep_on(Tp<T> a, Tp<T> b) const {
            return _imm_deps.count(a) && _imm_deps.at(a).count(b) > 0;
        }

        // Checks for immediate dependencies in either direction.
        virtual bool is_imm_dep(Tp<T> a, Tp<T> b) const {
            return is_imm_dep_on(a, b) || is_imm_dep_on(b, a);
        }

        // Check whether eq 'a' depends on 'b'.
        virtual bool is_dep_on(Tp<T> a, Tp<T> b) const {
            assert(_done);
            return _full_deps.count(a) && _full_deps.at(a).count(b) > 0;
        }

        // Checks for dependencies in either direction.
        virtual bool is_dep(Tp<T> a, Tp<T> b) const {
            return is_dep_on(a, b) || is_dep_on(b, a);
        }

        // Get all the objects that 'a' depends on.
        virtual const TpSet<T>& get_imm_deps_on(Tp<T> a) const {
            if (_imm_deps.count(a) == 0)
                return _empty;
            return _imm_deps.at(a);
        }
        virtual const TpSet<T>& get_all_deps_on(Tp<T> a) const {
            assert(_done);
            if (_full_deps.count(a) == 0)
                return _empty;
            return _full_deps.at(a);
        }

        // Visit 'a' and all its dependencies.
        // At each dep node 'b' in graph, 'visitor(b, path)' is called,
        // where 'path' contains all nodes from 'a' thru 'b' in dep order.
        virtual void visit_deps(Tp<T> a,
                                std::function<void (Tp<T> b,
                                                    TpList<T>& path)> visitor) const {
            _visit_deps(a, visitor, NULL);
        }

        // Print deps for debugging. 'T' must implement get_descr().
        virtual void print_deps(ostream& os) const {
            os << "Dependencies within " << _all.size() << " objects:\n";
            for (auto& a : _all) {
                os << " For " << a->get_descr() << ":\n";
                visit_deps(a, [&](Tp<T> b, TpList<T>& path) {
                                  if (a == b)
                                      os << "  depends on self";
                                  else
                                      os << "  depends on " << b->get_descr();
                                  os << " w/path of length " << path.size() << endl;
                              });
            }
        }

        // Do recursive analysis to find transitive closure.
        // https://en.wikipedia.org/wiki/Transitive_closure#In_graph_theory
        virtual void find_all_deps() {
            if (_done)
                return;
            for (auto a : _all)
                if (_full_deps.count(a) == 0)
                    visit_deps
                        (a, [&](Tp<T> b, TpList<T>& path) {

                                // Walk path from ee to b.
                                // Every 'c' in 'path' before 'b' depends on 'b'.
                                for (auto c : path)
                                    if (c != b)
                                        _full_deps[c].insert(b);
                            });
            _done = true;
        }
    };

    // A set of objects that have inter-dependencies.  Class 'T' must
    // implement 'clone()' that returns a 'shared_ptr<T>', 'is_scratch()',
    // 'get_descr()'.  Any 2 objects of class 'T' (at different addrs) are
    // considered different, even if they are identical.
    template <typename T>
    class DepGroup {

        // Example:
        // eq1: scr(x) EQUALS u(t,x+1);
        // eq2: u(t+1,x) EQUALS scr(x+2);
        // Direct deps: eq2 -> eq1(s).
        // eq1 is scratch child of eq2.

        // Example:
        // eq1: scr1(x) EQUALS u(t,x+1);
        // eq2: scr2(x) EQUALS scr1(x+2);
        // eq3: u(t+1,x) EQUALS scr2(x+4);
        // Direct deps: eq3 -> eq2(s) -> eq1(s).
        // eq1 and eq2 are scratch children of eq3.
        
    private:

        // Objs in this group.
        TpList<T> _all;

        // Dependencies between all objs.
        Deps<T> _deps;

        // Dependencies on scratch objs.
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
        virtual void add_item(Tp<T> p) {
            _all.insert(p);
        }
        virtual const TpList<T>& get_all() const {
            return _all;
        }
        virtual int get_num() const {
            return _all.size();
        }

        // Get the deps.
        virtual const Deps<T>& get_deps() const {
            return _deps;
        }
        virtual const TpSetMap<T>& get_imm_deps() const {
            return _deps.get_imm_deps();
        }
        virtual const TpSet<T>& get_all_deps_on(Tp<T> p) const {
            return _deps.get_all_deps_on(p);
        }
        virtual const TpSet<T>& get_imm_deps_on(Tp<T> p) const {
            return _deps.get_imm_deps_on(p);
        }

        // Get the scratch deps.
        virtual const Deps<T>& get_scratch_deps() const {
            return _scratches;
        }
        virtual const TpSet<T>& get_all_scratch_deps_on(Tp<T> p) const {
            return _scratches.get_all_deps_on(p);
        }
        virtual const TpSet<T>& get_imm_scratch_deps_on(Tp<T> p) const {
            return _scratches.get_imm_deps_on(p);
        }
        
        // Set deps.
        // Adds items if not already added.
        virtual void set_imm_dep_on(Tp<T> a, Tp<T> b) {
            add_item(a);
            add_item(b);
            _deps.set_imm_dep_on(a, b);
            if (b->is_scratch())
                _scratches.set_imm_dep_on(a, b);
            if (a == b)
                THROW_YASK_EXCEPTION("immediate dependency of '" + a->get_descr() +
                                     " on itself");
        }

        // Clear all deps.
        virtual void clear_deps() {
            _deps.clear_deps();
            _scratches.clear_deps();
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

                        // Dependence on self?
                        if (oi == oj)
                            THROW_YASK_EXCEPTION("indirect dependency of " +
                                                 oi->get_descr() + " on itself");

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

            // Two passes:
            // 0: set immediate deps and find indirect deps.
            // 1: check again for circular dependencies.

            for (int pass : { 0, 1 }) {

                // All T objs in this.
                for (auto& oi : _all) {

                    // All other T objs in this.
                    for (auto& oj : _all) {

                        // Don't compare to self.
                        if (oi == oj) continue;

                        if (pass == 1) {
                            
                            // Look for 2-way dep.
                            if (_deps.is_dep_on(oi, oj) &&
                                _deps.is_dep_on(oj, oi)) {
                                THROW_YASK_EXCEPTION("circular dependency between " +
                                                     oi->get_descr() + " and " +
                                                     oj->get_descr());
                            }
                        }
                        
                        // All Tf objs in 'oi'.
                        for (auto& foi : oi->get_items()) {

                            // All Tf objs in 'oj'.
                            for (auto& foj : oj->get_items()) {

                                if (pass == 0) {

                                    // Copy deps from 'full', i.e.,
                                    // if 'foi' is dep on 'foj',
                                    // then 'oi' is dep on 'oj'.
                                    if (fdeps.is_imm_dep_on(foi, foj))
                                        _deps.set_imm_dep_on(oi, oj);
                                    if (fscrs.is_imm_dep_on(foi, foj))
                                        _scratches.set_imm_dep_on(oi, oj);
                                }
                            }
                        }
                    }
                }
                if (pass == 0)
                    find_all_deps();
            } // passes.
        }
    };


    // A set of logical vars and related dependency data.
    class LogicalVars : public DepGroup<LogicalVar> {
    protected:
        Solution* _soln;
        map<std::string, LogicalVarPtr> _log_vars;

    public:
        LogicalVars(Solution* soln) : _soln(soln) { }
        virtual ~LogicalVars() { }

        LogicalVarPtr add_var_slice(VarPoint* vp) {

            // Already exists?
            auto descr = vp->make_logical_var_str();
            auto it = _log_vars.find(descr);
            if (it != _log_vars.end())
                return it->second;
            
            auto p = make_shared<LogicalVar>(_soln, vp);
            add_item(p);
            _log_vars[descr] = p;
            return p;
        }

        void print_info() const;
     };

    // A set of equations and related dependency data.
    class Eqs : public DepGroup<EqualsExpr> {
    protected:
        Solution* _soln;

    public:
        Eqs(Solution* soln) : _soln(soln) { }
        virtual ~Eqs() { }

        // Visit all equations.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& ep : get_all()) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.
        virtual void analyze_eqs();

        // Determine which var points can be vectorized.
        virtual void analyze_vec();

        // Determine how var points are accessed in a loop.
        virtual void analyze_loop();

        // Update var access stats.
        virtual void update_var_stats();

    };

    // A collection that holds various independent eqs.
    class EqLot {
    protected:
        Solution* _soln;
        
    private:
        EqList _eqs;            // all equations in this lot.
        Vars _out_vars;         // vars updated by _eqs.
        Vars _in_vars;          // vars read from by _eqs.
        bool _is_scratch = false; // true if _eqs update temp var(s).

    public:

        // Parts of the name.
        // TODO: move these into protected section and make accessors.
        string base_name;            // base name of this bundle.
        int index;                  // index to distinguish repeated names.

        // Ctor.
        EqLot(Solution* soln, bool is_scratch) :
            _soln(soln),
            _out_vars(soln),
            _in_vars(soln),
            _is_scratch(is_scratch) { }
        virtual ~EqLot() {}

        // Remove data.
        virtual void clear() {
            _eqs.clear();
            _out_vars.clear();
            _in_vars.clear();
        }

        // Add/remove an eq.
        virtual void add_eq(Eq ee);
        virtual void remove_eq(Eq ee);
        
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
    // will be combined into a single code block.
    class EqBundle : public EqLot {
 
    public:
        EqBundle(Solution* soln, bool is_scratch) :
            EqLot(soln, is_scratch) { }
        virtual ~EqBundle() { }

        // TODO: move these into protected section and make accessors.

        // Common conditions.
        bool_expr_ptr cond;
        bool_expr_ptr step_cond;

        // Common step expr.
        num_expr_ptr step_expr;

        // Create a copy containing clones of the equations.
        virtual shared_ptr<EqBundle> clone() const {
            auto p = make_shared<EqBundle>(_soln, is_scratch());

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->clear();
            for (auto& i : get_eqs())
                p->add_eq(i->clone());

            return p;
        }

        // Get a string description.
        virtual string get_descr(bool show_cond = true,
                                 string quote = "'") const;

        // Get the list of all equations.
        virtual const EqList& get_items() const {
            return get_eqs();
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
        virtual void replicate_eqs_in_cluster();
    };

    // Container for multiple equation bundles.
    class EqBundles : public DepGroup<EqBundle> {
    protected:
        Solution* _soln;

        string _base_name = "bundle";

        // Track vars that are updated.
        Vars _out_vars;

        // Bundle index;
        int _idx = 0;

        // Track equations that have been added already.
        set<Eq> _eqs_in_bundles;

        // Add 'eq' (subset of 'all_eqs') to an existing eq-bundle if
        // possible.  If not possible, create a new bundle and add 'eqs' to
        // it. The index will be incremented if a new bundle is created.
        // Returns whether a new bundle was created.
        virtual bool add_eq_to_bundle(Eq& eq);

    public:
        EqBundles(Solution* soln) :
            _soln(soln),
            _out_vars(soln) { }
        virtual ~EqBundles() { }

       // Separate a set of equations into eq_bundles based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "eq_bundle1=foo,eq_bundle2=bar".
        // In this example, all eqs updating var names containing 'foo' go in eq_bundle1,
        // all eqs updating var names containing 'bar' go in eq_bundle2, and
        // each remaining eq goes into a separate eq_bundle.
        void make_eq_bundles();

        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all eq_bundles.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& eg : get_all())
                eg->visit_eqs(ev);
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicate_eqs_in_cluster() {
            for (auto& eg : get_all())
                eg->replicate_eqs_in_cluster();
        }

        // Print stats for the equation(s) in all bundles.
        virtual void print_stats(const string& msg);

        // Apply optimizations requested in settings.
        virtual void optimize_eq_bundles(const string& descr);
    };
    typedef shared_ptr<EqBundle> EqBundlePtr;

    // A list of unique equation bundles.
    typedef vector_set<EqBundlePtr> EqBundleList;

    // A named equation stage, which contains one or more equation bundles.
    // All equations in a stage do not need to have the same domain
    // condition, but non-scratch eqs must have the same step condition.
    // Equations in a stage must not have inter-dependencies because they
    // may be run in parallel or in any order on any sub-domain.
    class EqStage : public EqLot {
    protected:
        EqBundleList _bundles;  // bundles in this stage.

    public:

        // Common condition.
        bool_expr_ptr step_cond;

        // Ctor.
        EqStage(Solution* soln, bool is_scratch) :
            EqLot(soln, is_scratch) { }
        virtual ~EqStage() { }

        virtual void clear() override {
            EqLot::clear();
            _bundles.clear();
        }

        // Create a copy containing clones of the bundles.
        virtual shared_ptr<EqStage> clone() const {
            auto p = make_shared<EqStage>(_soln, is_scratch());

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->clear();
            for (auto& i : _bundles)
                p->add_bundle(i->clone());

            return p;
        }

        // Get a string description.
        virtual string get_descr(string quote = "'") const;

        // Add/remove a bundle to this stage.
        virtual void add_bundle(EqBundlePtr ee);
        virtual void remove_bundle(EqBundlePtr ee);

        // Get the list of all bundles
        virtual EqBundleList& get_bundles() {
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
    typedef shared_ptr<EqStage> EqStagePtr;

    // Container for multiple equation stages.
    class EqStages : public DepGroup<EqStage> {
    protected:
        Solution* _soln;
        
        string _base_name = "stage";

        // Stage index.
        int _idx = 0;

        // Track vars that are updated.
        Vars _out_vars;

        // Track bundles that have been added already.
        set<EqBundlePtr> _bundles_in_stages;

        // Add 'bps', a subset of 'all_bundles'. Create new stage if needed.
        // Returns whether a new stage was created.
        bool add_bundles_to_stage(EqBundles& all_bundles,
                                  EqBundleList& bps,
                                  bool var_grouping,
                                  bool logical_var_grouping);

    public:
        EqStages(Solution* soln) :
            _soln(soln),
            _out_vars(soln) { }
        virtual ~EqStages() { }

        // Separate bundles into stages.
        void make_stages(EqBundles& bundles);

        // Get all output vars.
        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all stages.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& bp : get_all())
                bp->visit_eqs(ev);
        }

        // Find halos needed for each var.
        virtual void calc_halos(EqBundles& all_bundles);

    }; // EqStages.

} // namespace yask.

