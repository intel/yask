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

///////// Classes for equations, equation bundles, and bundle packs. ////////////

#ifndef EQS_HPP
#define EQS_HPP

#include "Expr.hpp"
#include "Grid.hpp"

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

        // Recursive helper for visitDeps().
        virtual void _visitDeps(Tp a,
                                std::function<void (Tp b, TpList& path)> visitor,
                                TpList* seen) const {

            // Already visited, i.e., a loop?
            bool was_seen = (seen && seen->count(a));
            if (was_seen)
                return;

            // Add 'a' to copy of path.
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
                    _visitDeps(b, visitor, &seen1);
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
        virtual void visitDeps(Tp a,
                               std::function<void (Tp b,
                                                   TpList& path)> visitor) const {
            _visitDeps(a, visitor, NULL);
        }

        // Print deps. 'T' must implement getDescr().
        virtual void printDeps(ostream& os) const {
            os << "Dependencies within " << _all.size() << " objects:\n";
            for (auto& a : _all) {
                os << " For " << a->getDescr() << ":\n";
                visitDeps(a, [&](Tp b, TpList& path) {
                        if (a == b)
                            os << "  depends on self";
                        else
                            os << "  depends on " << b->getDescr();
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
                    visitDeps(a, [&](Tp b, TpList& path) {

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
        virtual void addItem(Tp p) {
            _all.insert(p);
        }
        virtual const TpList& getAll() const {
            return _all;
        }
        virtual int getNum() const {
            return _all.size();
        }

        // Get the deps.
        virtual const Deps<T>& getDeps() const {
            return _deps;
        }
        virtual Deps<T>& getDeps() {
            return _deps;
        }
        virtual const TpSet& getDeps(Tp p) const {
            return _deps.get_deps_on(p);
        }

        // Get the scratch deps.
        virtual const Deps<T>& getScratchDeps() const {
            return _scratches;
        }
        virtual Deps<T>& getScratchDeps() {
            return _scratches;
        }
        virtual const TpSet& getScratchDeps(Tp p) const {
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

            // Scan from beginning to end.
            for (size_t i = 0; i < _all.size(); i++) {
                auto& oi = _all.at(i);

                // Repeat until no dependent found.
                bool done = false;
                while (!done) {

                    // Does obj[i] depend on any obj after it?
                    for (size_t j = i+1; j < _all.size(); j++) {
                        auto& oj = _all.at(j);

                        // Must swap if dependent.
                        if (_deps.is_dep_on(oi, oj)) {

                            // Error if also back-dep.
                            if (_deps.is_dep_on(oj, oi)) {
                                THROW_YASK_EXCEPTION("Error: circular dependency between " +
                                                     oi->getDescr() + " and " +
                                                     oj->getDescr());
                            }

                            // Swap them.
                            auto temp = oi;
                            oi = oj;
                            oj = temp;

                            // Start over at index i.
                            done = false;
                            break;
                        }
                    }
                    done = true;
                }
            }
        }

        // Copy dependencies from the 'full' graph to this
        // condensed graph.
        // Class 'T' must implement 'getItems()', which returns
        // an iteratable container of 'Tf' types.
        // See https://en.wikipedia.org/wiki/Directed_acyclic_graph.
        template <typename Tf>
        void inherit_deps_from(const DepGroup<Tf>& full) {

            // Deps between Tf objs.
            auto& fdeps = full.getDeps();
            auto& fscrs = full.getScratchDeps();

            // All T objs in this.
            for (auto& oi : _all) {

                // All other T objs in this.
                for (auto& oj : _all) {

                    // Don't compare to self.
                    if (oi == oj) continue;

                    // All Tf objs in 'oi'.
                    for (auto& foi : oi->getItems()) {

                        // All Tf objs in 'oj'.
                        for (auto& foj : oj->getItems()) {

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
    typedef vector_set<EqualsExprPtr> EqList;

    // A set of equations and related dependency data.
    class Eqs : public DepGroup<EqualsExpr> {

    public:

        // Visit all equations.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _all) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.
        virtual void analyzeEqs(CompilerSettings& settings,
                                Dimensions& dims,
                                std::ostream& os);

        // Determine which grid points can be vectorized.
        virtual void analyzeVec(const Dimensions& dims);

        // Determine how grid points are accessed in a loop.
        virtual void analyzeLoop(const Dimensions& dims);

        // Update grid access stats.
        virtual void updateGridStats();

        // Find scratch-grid eqs needed for each non-scratch eq.
        virtual void analyzeScratch();
    };

    // A collection that holds various independent eqs.
    class EqLot {
    protected:
        EqList _eqs;            // all equations.
        Grids _outGrids;        // grids updated by _eqs.
        Grids _inGrids;         // grids read from by _eqs.
        bool _isScratch = false; // true if _eqs update temp grid(s).

    public:

        // Parts of the name.
        // TODO: move these into protected section and make accessors.
        string baseName;            // base name of this bundle.
        int index;                  // index to distinguish repeated names.

        // Ctor.
        EqLot(bool is_scratch) : _isScratch(is_scratch) { }
        virtual ~EqLot() {}

        // Get all eqs.
        virtual const EqList& getEqs() const {
            return _eqs;
        }

        // Visit all the equations.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
                ep->accept(ev);
            }
        }

        // Get the full name.
        virtual string getName() const;

        // Get number of equations.
        virtual int getNumEqs() const {
            return _eqs.size();
        }

        // Updating temp vars?
        virtual bool isScratch() const { return _isScratch; }

        // Get grids output and input.
        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }
        virtual const Grids& getInputGrids() const {
            return _inGrids;
        }

        // Print stats for the equation(s).
        virtual void printStats(ostream& os, const string& msg);
    };

    // A named equation bundle, which contains one or more grid-update
    // equations.  All equations in a bundle must have the same condition.
    // Equations in a bundle must not have inter-dependencies because they
    // will be combined into a single expression.
    class EqBundle : public EqLot {
    protected:
        const Dimensions* _dims = 0;

    public:

        // TODO: move these into protected section and make accessors.

        // Common condition.
        BoolExprPtr cond;

        // Common step expr.
        NumExprPtr step_expr;

        // Create a copy containing clones of the equations.
        virtual shared_ptr<EqBundle> clone() const {
            auto p = make_shared<EqBundle>(*_dims, _isScratch);

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
        virtual string getDescr(bool show_cond = true,
                                string quote = "'") const;

        // Add an equation to this bundle.
        virtual void addEq(EqualsExprPtr ee);

        // Get the list of all equations.
        virtual const EqList& getItems() const {
            return _eqs;
        }

        // Visit the condition.
        // Return true if there was one to visit.
        virtual bool visitCond(ExprVisitor* ev) {
            if (cond.get()) {
                cond->accept(ev);
                return true;
            }
            return false;
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims);
    };

    // Container for multiple equation bundles.
    class EqBundles : public DepGroup<EqBundle> {
    protected:

        // Copy of some global data.
        string _basename_default;
        Dimensions* _dims = 0;

        // Track grids that are updated.
        Grids _outGrids;

        // Map to track indices per eq-bundle name.
        map<string, int> _indices;

        // Track equations that have been added already.
        set<EqualsExprPtr> _eqs_in_bundles;

        // Add 'eq' from 'eqs' to eq-bundle with 'baseName'
        // unless already added or illegal.  The corresponding index in
        // '_indices' will be incremented if a new bundle is created.
        // Returns whether a new bundle was created.
        virtual bool addEqToBundle(Eqs& eqs,
                                   EqualsExprPtr eq,
                                   const string& baseName,
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

        // Separate a set of equations into eqBundles based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "eqBundle1=foo,eqBundle2=bar".
        // In this example, all eqs updating grid names containing 'foo' go in eqBundle1,
        // all eqs updating grid names containing 'bar' go in eqBundle2, and
        // each remaining eq goes into a separate eqBundle.
        void makeEqBundles(Eqs& eqs,
                           const CompilerSettings& settings,
                           std::ostream& os);

        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }

        // Visit all the equations in all eqBundles.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& eg : _all)
                eg->visitEqs(ev);
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims) {
            for (auto& eg : _all)
                eg->replicateEqsInCluster(dims);
        }

        // Print stats for the equation(s) in all bundles.
        virtual void printStats(ostream& os, const string& msg);

        // Apply optimizations requested in settings.
        virtual void optimizeEqBundles(CompilerSettings& settings,
                                       const string& descr,
                                       bool printSets,
                                       ostream& os);

        // Adjust scratch-grid halos as needed.
        virtual void adjustScratchHalos();
    };

    typedef shared_ptr<EqBundle> EqBundlePtr;

    // A list of unique equation bundles.
    typedef vector_set<EqBundlePtr> EqBundleList;

    // A named equation bundle pack, which contains one or more equation
    // bundles.  All equations in a pack do not need to have the same condition.
    // Equations in a pack must not have inter-dependencies because they
    // may be run in parallel or in any order on any sub-domain.
    class EqBundlePack : public EqLot {
    protected:
        EqBundleList _bundles;  // bundles in this pack.

    public:

        // Ctor.
        EqBundlePack(bool is_scratch) :
            EqLot(is_scratch) { }
        virtual ~EqBundlePack() { }

        // Create a copy containing clones of the bundles.
        virtual shared_ptr<EqBundlePack> clone() const {
            auto p = make_shared<EqBundlePack>(_isScratch);

            // Shallow copy.
            *p = *this;

            // Delete copied eqs and replace w/clones.
            p->_eqs.clear();
            for (auto& i : _bundles)
                p->_bundles.insert(i->clone());

            return p;
        }

        // Get a string description.
        virtual string getDescr(string quote = "'") const;

        // Add a bundle to this pack.
        virtual void addBundle(EqBundlePtr ee);

        // Get the list of all bundles
        virtual const EqBundleList& getBundles() const {
            return _bundles;
        }
        virtual const EqBundleList& getItems() const {
            return _bundles;
        }

    };

    // Container for multiple equation bundle packs.
    class EqBundlePacks : public DepGroup<EqBundlePack> {
    protected:
        string _baseName = "stencil_pack";

        // Bundle index.
        int _idx = 0;

        // Track grids that are updated.
        Grids _outGrids;

        // Track bundles that have been added already.
        set<EqBundlePtr> _bundles_in_packs;

        // Add 'bp' from 'allBundles'. Create new pack if needed.  Returns
        // whether a new pack was created.
        bool addBundleToPack(EqBundles& allBundles,
                             EqBundlePtr bp);

    public:

        // Separate bundles into packs.
        void makePacks(EqBundles& bundles,
                       std::ostream& os);

        // Get all output grids.
        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }

        // Visit all the equations in all packs.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& bp : _all)
                bp->visitEqs(ev);
        }
    };

} // namespace yask.

#endif
