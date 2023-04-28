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

///////// Classes for equations, parts, and stages. ////////////

#pragma once

#include "Expr.hpp"
#include "VarPoint.hpp"
#include "Settings.hpp"

using namespace std;

//#define DEBUG_DEPS

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

        // Recursive helper for visit_dep_paths().
        virtual void _visit_dep_paths(Tp<T> a,
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

                // loop thru deps of 'a', i.e., 'a' deps on each 'b'.
                for (auto b : adeps) {

                    // Recurse to deps of 'b'.
                    _visit_dep_paths(b, visitor, &seen1);
                }
            }
        }

        // Recursive helper for dfs_visit_deps().
        virtual void _dfs_visit_deps(Tp<T> a, bool preorder,
                                     std::function<void (Tp<T> b)> visitor,
                                     TpSet<T>& visited, TpSet<T>& finished) const {

            // Done?
            if (finished.count(a))
                return;

            // Cycle?
            if (visited.count(a)) {
                THROW_YASK_EXCEPTION("circular dependency on '" +
                                     a->get_descr() + "'");
            }
            visited.insert(a);

            // Callback.
            if (preorder)
                visitor(a);
            
            // any dependencies?
            if (_imm_deps.count(a)) {
                auto& adeps = _imm_deps.at(a);

                // loop thru deps of 'a', i.e., 'a' deps on each 'b'.
                for (auto b : adeps) {

                    // Recurse.
                    _dfs_visit_deps(b, preorder, visitor, visited, finished);

                }
            }

            // Callback.
            if (!preorder)
                visitor(a);
            
            // Done.
            finished.insert(a);
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

        // Visit 'a' and all its dependencies along every possible path.  At
        // each dep node 'b' in graph, 'visitor(b, path)' is called, where
        // 'path' contains all nodes from 'a' thru 'b' in dep order.
        // Warning: use only for debug--dependency tree may cause a
        // combinatorial explosion in number of paths: millions have been
        // seen in example stencils.
        virtual void visit_dep_paths(Tp<T> a,
                                     std::function<void (Tp<T> b,
                                                         TpList<T>& path)> visitor) const {
            _visit_dep_paths(a, visitor, NULL);
        }

        // Visit 'a' and dependencies 'b' of 'a' using a depth-first-search
        // with pre-ordering, i.e., visit nodes before their children or
        // post-ordering, i.e., visit children before their parents.
        void dfs_visit_deps(Tp<T> a, bool preorder,
                            std::function<void (Tp<T> b)> visitor) const {
            TpSet<T> visited, finished;
            _dfs_visit_deps(a, preorder, visitor, visited, finished);
        }

        // Print deps for debugging. 'T' must implement get_descr().
        void print_deps(ostream& os, bool preorder=true) const {
            os << "Dependencies within " << _all.size() << " objects:\n";
            for (auto& a : _all) {
                os << " from " << a->get_descr() << ":\n";
                dfs_visit_deps
                    (a, preorder,
                     [&](Tp<T> b) {
                         if (b != a)
                             os << "  " << b->get_descr() << endl;
                     });
            }
        }

        // Print dep paths for debugging. 'T' must implement get_descr().
        virtual void print_dep_paths(ostream& os) const {
            os << "Dependencies within " << _all.size() << " objects:\n";
            for (auto& a : _all) {
                os << " from " << a->get_descr() << ":\n";
                visit_dep_paths
                    (a, [&](Tp<T> b, TpList<T>& path) {
                            os << "  path (len " << path.size() << "):";
                            for (auto c : path)
                                os << " " << c->get_descr();
                            os << endl;
                        });
            }
        }

        // Do recursive analysis to find transitive closure.
        // https://en.wikipedia.org/wiki/Transitive_closure#In_graph_theory
        virtual void find_all_deps() {
            if (_done)
                return;

            #ifdef DEBUG_DEPS
            cout << "** find_all_deps among " << _all.size() << " objs...\n"
                "*** imm_deps has " << _imm_deps.size() << " entries w/sizes:";
            for (auto& a : _imm_deps)
                cout << " " << a.second.size();
            cout << endl;
            #endif
            
            for (auto a : _all) {
                #ifdef DEBUG_DEPS
                cout << "*** visiting deps of " << a->get_descr() << endl;
                int nd = 0;
                #endif

                // Visit every node 'b' in the imm-dep tree starting at 'a'.
                dfs_visit_deps
                    (a, true,
                     [&](Tp<T> b) {

                         // 'a' depends on 'b', directly or indirectly.
                         if (a != b)
                             _full_deps[a].insert(b);

                         #ifdef DEBUG_DEPS
                         nd++;
                         #endif
                     });
                #ifdef DEBUG_DEPS
                cout << "*** visited " << nd << " nodes\n";
                #endif
            }
            _done = true;
            #ifdef DEBUG_DEPS
            cout << "*** done finding all deps on " << _all.size() << " objs\n";
            #endif
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
        // NB: There may be objs in _all that are not in _deps.
        Deps<T> _deps;

        // Dependencies on scratch objs.
        // Entries will be a subset of _deps.
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
                THROW_YASK_EXCEPTION("circular dependency on '" +
                                     a->get_descr() + "'");
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

        // Reorder based on dependencies, i.e., topological sort.  Objs will
        // be sorted such that evaluations that need to be done first will
        // be at the beginning of the list.
        virtual void topo_sort() {

            // No need to sort less than two things.
            if (_all.size() <= 1)
                return;

            // Want to keep original order as much as possible.  Only
            // reorder if dependencies are in conflict.  NB: This also is
            // probably not as efficient as Kuhn's algorithm, but not bad,
            // and simpler. Depends on having transitive closure first.
            find_all_deps();
                
            // Scan obj list from beginning to next-to-last.
            for (size_t i = 0; i < _all.size() - 1; i++) {

                // Repeat until no dependent found.
                bool done = false;
                while (!done) {

                    // Does obj[i] depend on any obj after it?
                    auto& oi = _all.at(i);
                    done = true;

                    // Scan from i+1 to last.
                    for (size_t j = i+1; j < _all.size(); j++) {
                        auto& oj = _all.at(j);

                        // Found again?
                        if (oi == oj)
                            THROW_YASK_EXCEPTION("internal error: '" +
                                                 oi->get_descr() +
                                                 "' in dependency graph multiple times");

                        // Must swap if dependent.
                        if (_deps.is_dep_on(oi, oj)) {

                            // Error if also back-dep.
                            if (_deps.is_dep_on(oj, oi)) {
                                THROW_YASK_EXCEPTION("circular dependency on '" +
                                                     oi->get_descr() + "' via '" +
                                                     oj->get_descr() + "'");
                            }

                            // Swap them.
                            _all.swap(i, j);

                            // Start over at index i.
                            // This is safe because we're not moving anything
                            // before i.
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

                            if (foi == foj)
                                continue;

                            // Copy deps from 'full', i.e., if 'foi' is dep
                            // on 'foj', then 'oi' is dep on 'oj'.  This may
                            // create a circular dependency in 'this' that
                            // wasn't in 'full'.
                            if (fdeps.is_imm_dep_on(foi, foj))
                                _deps.set_imm_dep_on(oi, oj);
                            if (fscrs.is_imm_dep_on(foi, foj))
                                _scratches.set_imm_dep_on(oi, oj);
                        }
                    }
                }
            }

            // Find indirect deps.
            find_all_deps();

        } // inherit_deps_from().
    };


    // A set of logical vars and related dependency data.
    class LogicalVars : public DepGroup<LogicalVar> {
    protected:
        Solution* _soln;
        map<std::string, LogicalVarPtr> _log_vars;

    public:
        LogicalVars(Solution* soln) : _soln(soln) { }
        virtual ~LogicalVars() { }

        LogicalVarPtr find_var_slice(VarPoint* vp) {

            // Already exists?
            auto descr = vp->make_logical_var_str();
            auto it = _log_vars.find(descr);
            if (it != _log_vars.end())
                return it->second;

            return nullptr;
        }

        LogicalVarPtr add_var_slice(VarPoint* vp) {

            // Already exists?
            auto lvp = find_var_slice(vp);
            if (lvp)
                return lvp;

            // Make new one.
            auto descr = vp->make_logical_var_str();
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
        string base_name;            // base name of this part.
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

    // A named solution part, which contains one or more var-update
    // equations.  All equations in a part must have the same conditions.
    // Equations in a part must not have inter-dependencies because they
    // will be combined into a single code block.
    class Part : public EqLot {
 
    public:
        Part(Solution* soln, bool is_scratch) :
            EqLot(soln, is_scratch) { }
        virtual ~Part() { }

        // TODO: move these into protected section and make accessors.

        // Common conditions.
        bool_expr_ptr cond;
        bool_expr_ptr step_cond;

        // Common step expr.
        num_expr_ptr step_expr;

        // Create a copy containing clones of the equations.
        virtual Tp<Part> clone() const {
            auto p = make_shared<Part>(_soln, is_scratch());

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
    };
    typedef Tp<Part> PartPtr;
    typedef TpList<Part> PartList;
    typedef TpSet<Part> PartSet;

    // Container for multiple equation parts.
    class Parts : public DepGroup<Part> {
    protected:
        Solution* _soln;

        string _base_name = "part";

        // Track vars that are updated.
        Vars _out_vars;

        // part index;
        int _idx = 1;

        // Track equations that have been added already.
        set<Eq> _eqs_in_parts;

        // Add 'eq' (subset of 'all_eqs') to an existing part if
        // possible.  If not possible, create a new part and add 'eqs' to
        // it. The index will be incremented if a new part is created.
        // Returns whether a new part was created.
        virtual bool add_eq_to_part(Eq& eq);

    public:
        Parts(Solution* soln) :
            _soln(soln),
            _out_vars(soln) { }
        virtual ~Parts() { }

        // Separate a set of equations into parts based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "part1=foo,part2=bar".
        // In this example, all eqs updating var names containing 'foo' go in part1,
        // all eqs updating var names containing 'bar' go in part2, and
        // each remaining eq goes into a separate part.
        void make_parts();

        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all parts.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& eg : get_all())
                eg->visit_eqs(ev);
        }

        // Print stats for the equation(s) in all parts.
        virtual void print_stats(const string& msg);

        // Apply optimizations requested in settings.
        virtual void optimize_parts(const string& descr);
    };

    // A named equation stage, which contains one or more equation parts.
    // Equations in a stage must not have inter-dependencies because they
    // may be run in parallel or in any order on any sub-domain.
    class Stage : public EqLot {
    protected:
        PartList _parts;  // parts in this stage.

    public:

        // Ctor.
        Stage(Solution* soln, bool is_scratch) :
            EqLot(soln, is_scratch) { }
        virtual ~Stage() { }

        virtual void clear() override {
            EqLot::clear();
            _parts.clear();
        }

        // Create a copy containing clones of the parts.
        virtual shared_ptr<Stage> clone() const {
            auto p = make_shared<Stage>(_soln, is_scratch());

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->clear();
            for (auto& i : _parts)
                p->add_part(i->clone());

            return p;
        }

        // Get a string description.
        virtual string get_descr(string quote = "'") const;

        // Add/remove a part to this stage.
        virtual void add_part(PartPtr ee);
        virtual void remove_part(PartPtr ee);

        // Get the list of all parts
        virtual PartList& get_parts() {
            return _parts;
        }
        virtual const PartList& get_items() const {
            return _parts;
        }
    };
    typedef Tp<Stage> StagePtr;
    typedef TpSet<Stage> StageSet;
    typedef TpList<Stage> StageList;

    // Container for multiple equation stages.
    class Stages : public DepGroup<Stage> {
    protected:
        Solution* _soln;
        
        string _base_name = "stage";

        // Stage index.
        int _idx = 1;

        // Track vars that are updated.
        Vars _out_vars;

        // Track parts that have been added already.
        PartSet _parts_in_stages;

        // Add 'pps', a subset of 'all_parts'. Create new stage if needed.
        // Returns whether a new stage was created.
        bool add_parts_to_stage(Parts& all_parts,
                                PartList& pps,
                                bool var_grouping,
                                bool logical_var_grouping);

    public:
        Stages(Solution* soln) :
            _soln(soln),
            _out_vars(soln) { }
        virtual ~Stages() { }

        // Separate parts into stages.
        void make_stages(Parts& parts);

        // Get all output vars.
        virtual const Vars& get_output_vars() const {
            return _out_vars;
        }

        // Visit all the equations in all stages.
        virtual void visit_eqs(ExprVisitor* ev) {
            for (auto& pp : get_all())
                pp->visit_eqs(ev);
        }

        // Find halos needed for each var.
        virtual void calc_halos(Parts& all_parts);

    }; // Stages.

} // namespace yask.

