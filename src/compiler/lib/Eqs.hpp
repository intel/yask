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

///////// Classes for equations and equation bundles ////////////

#ifndef EQS_HPP
#define EQS_HPP

#include "Expr.hpp"
#include "Grid.hpp"

using namespace std;

namespace yask {

    // Dependencies between equations.
    class EqDeps {

    public:
        // dep_map[A].count(B) > 0 => A depends on B.
        typedef unordered_set<EqualsExprPtr> EqSet;
        typedef unordered_map<EqualsExprPtr, EqSet> DepMap;
        typedef vector_set<EqualsExprPtr> EqVecSet;

    protected:
        DepMap _imm_deps;       // immediate deps, i.e., transitive reduction.
        DepMap _full_deps;      // transitive closure of _imm_deps.
        EqSet _all;             // all expressions.
        bool _done;             // indirect dependencies added?
    
        // Recursive helper for visitDeps().
        virtual void _visitDeps(EqualsExprPtr a,
                               std::function<void (EqualsExprPtr b, EqVecSet& path)> visitor,
                               EqVecSet* seen) const;

    public:

        EqDeps() : _done(false) {}
        virtual ~EqDeps() {}
    
        // Declare that eq a depends directly on b.
        virtual void set_imm_dep_on(EqualsExprPtr a, EqualsExprPtr b) {
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
            _done = false;
        }
    
        // Check whether eq a directly depends on b.
        virtual bool is_imm_dep_on(EqualsExprPtr a, EqualsExprPtr b) const {
            return _imm_deps.count(a) && _imm_deps.at(a).count(b) > 0;
        }
    
        // Checks for immediate dependencies in either direction.
        virtual bool is_imm_dep(EqualsExprPtr a, EqualsExprPtr b) const {
            return is_imm_dep_on(a, b) || is_imm_dep_on(b, a);
        }

        // Check whether eq a depends on b.
        virtual bool is_dep_on(EqualsExprPtr a, EqualsExprPtr b) const {
            assert(_done || _imm_deps.size() == 0);
            return _full_deps.count(a) && _full_deps.at(a).count(b) > 0;
        }
    
        // Checks for dependencies in either direction.
        virtual bool is_dep(EqualsExprPtr a, EqualsExprPtr b) const {
            return is_dep_on(a, b) || is_dep_on(b, a);
        }

        // Visit 'a' and all its immediate dependencies.
        // At each node 'b', 'visitor(b, path)' is called, where 'path' contains
        // all nodes from 'a' thru 'b'.
        virtual void visitDeps(EqualsExprPtr a,
                               std::function<void (EqualsExprPtr b, EqVecSet& path)> visitor) {
            _visitDeps(a, visitor, NULL);
        }
        
        // Does recursive analysis to find indirect dependencies from direct
        // ones.
        virtual void find_all_deps();
    };

    // A collection of deps by dep type.
    typedef map<DepType, EqDeps> EqDepMap;

    // A list of unique equation ptrs.
    typedef vector_set<EqualsExprPtr> EqList;

    // Map w/key = expr ptr, value = if-condition ptr.
    // We use this to simplify the process of replacing statements
    //  when an if-condition is encountered.
    // Example: key: grid(t,x)==grid(t,x+1); value: x>5;
    typedef map<EqualsExpr*, BoolExprPtr> CondMap;

    // A set of equations and related data.
    class Eqs {

    protected:
    
        // Equations(s) describing how values in this grid are computed.
        EqList _eqs;

        // Dependencies between all eqs.
        EqDepMap _eq_deps;

        // Dependencies through scratch grids.
        EqDeps::DepMap _scratch_deps;
        
    public:

        Eqs() {
            // Make sure map keys exist.
            for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1))
                _eq_deps[dt];
        }
        virtual ~Eqs() {}

        // Equation accessors.
        virtual void addEq(EqualsExprPtr ep) {
            _eqs.insert(ep);
            _scratch_deps[ep];
        }
        virtual const EqList& getEqs() const {
            return _eqs;
        }
        virtual int getNumEqs() const {
            return _eqs.size();
        }

        // Get all the deps.
        virtual const EqDepMap& getDeps() const {
            return _eq_deps;
        }
        virtual EqDepMap& getDeps() {
            return _eq_deps;
        }
        
        // Get the scratch-grid eqs that contribute to 'eq'.
        virtual const EqDeps::EqSet& getScratchDeps(EqualsExprPtr ep) const {
            return _scratch_deps.at(ep);
        }

        // Visit all equations.
        // Will NOT visit conditions.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.  If 'eq_deps' is
        // set, save dependencies between eqs in referent.
        virtual void analyzeEqs(CompilerSettings& settings,
                                Dimensions& dims,
                                std::ostream& os);

        // Determine which grid points can be vectorized.
        virtual void analyzeVec(const Dimensions& dims);

        // Determine how grid points are accessed in a loop.
        virtual void analyzeLoop(const Dimensions& dims);

        // Update grid access stats.
        virtual void updateGridStats();
    };

    // A named equation bundle, which contains one or more grid-update
    // equations.  All equations in a bundle must have the same condition.
    // Equations in a bundle should not have inter-dependencies because they
    // will be combined into a single expression.
    class EqBundle {
    protected:
        EqList _eqs; // expressions in this eqBundle (not including conditions).
        Grids _outGrids;          // grids updated by this eqBundle.
        Grids _inGrids;          // grids read from by this eqBundle.
        const Dimensions* _dims = 0;
        bool _isScratch = false; // true if updating temp grid(s).

        // Other eq-bundles that this bundle depends on. This means that an
        // equation in this bundle has a grid value on the RHS that appears in
        // the LHS of the dependency.
        map<DepType, set<string>> _imm_dep_on; // immediate deps.
        map<DepType, set<string>> _dep_on;     // immediate and indirect deps.
        set<string> _scratch_deps;             // scratch bundles needed for this bundle.

    public:

        // TODO: move these into protected section and make accessors.
        string baseName;            // base name of this eqBundle.
        int index;                  // index to distinguish repeated names.
        BoolExprPtr cond;           // condition (default is null).

        // Ctor.
        EqBundle(const Dimensions& dims, bool is_scratch) :
            _dims(&dims), _isScratch(is_scratch) {

            // Create empty map entries.
            for (DepType dt = DepType(0); dt < num_deps; dt = DepType(dt+1)) {
                _imm_dep_on[dt];
                _dep_on[dt];
            }
        }
        virtual ~EqBundle() {}

        // Add an equation to this bundle.
        virtual void addEq(EqualsExprPtr ee);
    
        // Visit all the equations.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
#ifdef DEBUG_EQ_BUNDLE
                cout << "EqBundle: visiting " << ep->makeQuotedStr() << endl;
#endif
                ep->accept(ev);
            }
        }

        // Get the list of all equations.
        // Does NOT return condition.
        virtual const EqList& getEqs() const {
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

        // Get the full name.
        virtual string getName() const;

        // Get a string description.
        virtual string getDescription(bool show_cond = true,
                                      string quote = "'") const;

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

        // Get whether this eq-bundle depends on eg2.
        // Must have already been set via checkDeps().
        virtual bool isImmDepOn(DepType dt, const EqBundle& eq2) const {
            return _imm_dep_on.at(dt).count(eq2.getName()) > 0;
        }
        virtual bool isDepOn(DepType dt, const EqBundle& eq2) const {
            return _dep_on.at(dt).count(eq2.getName()) > 0;
        }

        // Get dependencies on this eq-bundle.
        virtual const set<string>& getImmDeps(DepType dt) const {
            return _imm_dep_on.at(dt);
        }
        virtual const set<string>& getDeps(DepType dt) const {
            return _dep_on.at(dt);
        }

        // Get scratch-bundle dependencies.
        virtual const set<string>& getScratchDeps() const {
            return _scratch_deps;
        }
    
        // Check for and set dependencies on eg2.
        virtual void checkDeps(Eqs& allEqs, const EqBundle& eg2);

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims);
        
        // Print stats for the equation(s) in this bundle.
        virtual void printStats(ostream& os, const string& msg);
    };

    // Container for multiple equation bundles.
    class EqBundles : public vector<EqBundle> {
    protected:

        // Copy of some global data.
        string _basename_default;
        Dimensions* _dims = 0;

        // Track grids that are udpated.
        Grids _outGrids;

        // Map to track indices per eq-bundle name.
        map<string, int> _indices;

        // Track equations that have been added already.
        set<EqualsExprPtr> _eqs_in_bundles;
    
        // Add expression 'eq' from 'eqs' to eq-bundle with 'baseName'
        // unless alread added.  The corresponding index in '_indices' will be
        // incremented if a new bundle is created.
        // Returns whether a new bundle was created.
        virtual bool addExprToBundle(Eqs& eqs,
                                     EqualsExprPtr eq,
                                     const string& baseName,
                                     bool is_scratch);

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
                           const string& gridRegex,
                           const string& targets,
                           std::ostream& os);
        
        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }

        // Visit all the equations in all eqBundles.
        // This will not visit the conditions.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& eg : *this)
                eg.visitEqs(ev);
        }

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims) {
            for (auto& eg : *this)
                eg.replicateEqsInCluster(dims);
        }

        // Reorder bundles based on dependencies.
        virtual void sort(ostream& os);
    
        // Print a list of eqBundles.
        virtual void printInfo(ostream& os) const {
            os << "Identified stencil equation-bundles:" << endl;
            for (auto& eq : *this) {
                for (auto gp : eq.getOutputGrids()) {
                    string eqName = eq.getName();
                    os << "  Equation bundle '" << eqName << "' updates grid '" <<
                        gp->getName() << "'." << endl;
                }
            }
        }

        // Print stats for the equation(s) in all bundles.
        virtual void printStats(ostream& os, const string& msg);

        // Apply optimizations requested in settings.
        void optimizeEqBundles(CompilerSettings& settings,
                              const string& descr,
                              bool printSets,
                              ostream& os);
    };
    
} // namespace yask.
    
#endif
