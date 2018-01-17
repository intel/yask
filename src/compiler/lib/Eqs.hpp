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

///////// Classes for equations and equation groups ////////////

#ifndef EQS_HPP
#define EQS_HPP

#include "Expr.hpp"
#include "Grid.hpp"

using namespace std;

namespace yask {

    // Types of dependencies.
    // Must keep this consistent with list in stencil_calc.hpp.
    // TODO: make this common code w/kernel.
    enum DepType {
        certain_dep,
        possible_dep,
        num_deps
    };

    // Dependencies between equations.
    // eq_deps[A].count(B) > 0 => A depends on B.
    class EqDeps {
    protected:
        typedef unordered_map<EqualsExprPtr, unordered_set<EqualsExprPtr>> DepMap;
        typedef set<EqualsExprPtr> SeenSet;

        DepMap _deps;               // direct dependencies.
        DepMap _full_deps;          // direct and indirect dependencies.
        SeenSet _all;               // all expressions.
        bool _done;                 // indirect dependencies added?
    
        // Recursive search starting at 'a'.
        // Fill in _full_deps.
        virtual bool _analyze(EqualsExprPtr a, SeenSet* seen);
    
    public:

        EqDeps() : _done(false) {}
        virtual ~EqDeps() {}
    
        // Declare that eq a depends on b.
        virtual void set_dep_on(EqualsExprPtr a, EqualsExprPtr b) {
            _deps[a].insert(b);
            _all.insert(a);
            _all.insert(b);
            _done = false;
        }
    
        // Check whether eq a depends on b.
        virtual bool is_dep_on(EqualsExprPtr a, EqualsExprPtr b) const {
            assert(_done || _deps.size() == 0);
            return _full_deps.count(a) && _full_deps.at(a).count(b) > 0;
        }
    
        // Checks for dependencies in either direction.
        virtual bool is_dep(EqualsExprPtr a, EqualsExprPtr b) const {
            return is_dep_on(a, b) || is_dep_on(b, a);
        }

        // Does recursive analysis to turn all indirect dependencies to direct
        // ones.
        virtual void analyze() {
            if (_done)
                return;
            for (auto a : _all)
                if (_full_deps.count(a) == 0)
                    _analyze(a, NULL);
            _done = true;
        }
    };

    typedef map<DepType, EqDeps> EqDepMap;

    // A list of unique equation ptrs.
    typedef vector_set<EqualsExprPtr> EqList;

    // Map of expressions: key = expression ptr, value = if-condition ptr.
    // We use this to simplify the process of replacing statements
    //  when an if-condition is encountered.
    // Example: key: grid(t,x)==grid(t,x+1); value: x>5;
    typedef map<EqualsExpr*, BoolExprPtr> CondMap;

    // A set of equations and related data.
    class Eqs {

    protected:
    
        // Equations(s) describing how values in this grid are computed.
        EqList _eqs;          // just equations w/o conditions.
        CondMap _conds;       // map from equations to their conditions, if any.

    public:

        Eqs() {}
        virtual ~Eqs() {}

        // Equation accessors.
        virtual void addEq(EqualsExprPtr ep) {
            _eqs.insert(ep);
        }
        virtual void addCondEq(EqualsExprPtr ep, BoolExprPtr cond) {
            _eqs.insert(ep);
            _conds[ep.get()] = cond;
        }
        virtual const EqList& getEqs() const {
            return _eqs;
        }
        virtual int getNumEqs() const {
            return _eqs.size();
        }

        // Get the condition associated with an expression.
        // If there is no condition, a null pointer is returned.
        virtual const BoolExprPtr getCond(EqualsExprPtr ep) {
            return getCond(ep.get());
        }
        virtual const BoolExprPtr getCond(EqualsExpr* ep) {
            if (_conds.count(ep))
                return _conds.at(ep);
            else
                return nullptr;
        }

        // Visit all equations.
        // Will NOT visit conditions.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
                ep->accept(ev);
            }
        }

        // Find dependencies based on all eqs.  If 'eq_deps' is
        // set, save dependencies between eqs.
        virtual void findDeps(IntTuple& pts,
                              Dimensions& dims,
                              EqDepMap* eq_deps,
                              std::ostream& os);

        // Check for illegal dependencies in all equations.
        // Exit with error if any found.
        virtual void checkDeps(IntTuple& pts,
                               Dimensions& dims,
                               std::ostream& os) {
            findDeps(pts, dims, NULL, os);
        }

        // Determine which grid points can be vectorized.
        virtual void analyzeVec(const Dimensions& dims);

        // Determine how grid points are accessed in a loop.
        virtual void analyzeLoop(const Dimensions& dims);
    };

    // A named equation group, which contains one or more grid-update equations.
    // All equations in a group must have the same condition.
    // Equations should not have inter-dependencies because they will be
    // combined into a single expression.
    class EqGroup {
    protected:
        EqList _eqs; // expressions in this eqGroup (not including conditions).
        Grids _outGrids;          // grids updated by this eqGroup.
        Grids _inGrids;          // grids read from by this eqGroup.
        const Dimensions* _dims = 0;

        // Other eq-groups that this group depends on. This means that an
        // equation in this group has a grid value on the RHS that appears in
        // the LHS of the dependency.
        map<DepType, set<string>> _dep_on;

    public:
        string baseName;            // base name of this eqGroup.
        int index;                  // index to distinguish repeated names.
        BoolExprPtr cond;           // condition (default is null).

        // Ctor.
        EqGroup(const Dimensions& dims) : _dims(&dims) {

            // Create empty map entries.
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                _dep_on[dt];
            }
        }
        virtual ~EqGroup() {}

        // Add an equation.
        // If 'update_stats', update grid and halo data.
        virtual void addEq(EqualsExprPtr ee, bool update_stats = true);
    
        // Visit all the equations.
        virtual void visitEqs(ExprVisitor* ev) {
            for (auto& ep : _eqs) {
#ifdef DEBUG_EQ_GROUP
                cout << "EqGroup: visiting " << ep->makeQuotedStr() << endl;
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

        // Get grids output and input.
        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }
        virtual const Grids& getInputGrids() const {
            return _inGrids;
        }

        // Get whether this eq-group depends on eg2.
        // Must have already been set via setDepOn().
        virtual bool isDepOn(DepType dt, const EqGroup& eq2) const {
            return _dep_on.at(dt).count(eq2.getName()) > 0;
        }

        // Get dependencies on this eq-group.
        virtual const set<string>& getDeps(DepType dt) const {
            return _dep_on.at(dt);
        }
    
        // Set dependency on eg2 if this eq-group is dependent on it.
        // Return whether dependent.
        virtual bool setDepOn(DepType dt, EqDepMap& eq_deps, const EqGroup& eg2);

        // Replicate each equation at the non-zero offsets for
        // each vector in a cluster.
        virtual void replicateEqsInCluster(Dimensions& dims);
        
        // Print stats for the equation(s) in this group.
        virtual void printStats(ostream& os, const string& msg);
    };

    // Container for multiple equation groups.
    class EqGroups : public vector<EqGroup> {
    protected:

        // Copy of some global data.
        string _basename_default;
        Dimensions* _dims = 0;

        // Track grids that are udpated.
        Grids _outGrids;

        // Map to track indices per eq-group name.
        map<string, int> _indices;

        // Track equations that have been added already.
        set<EqualsExprPtr> _eqs_in_groups;
    
        // Add expression 'eq' with condition 'cond' to eq-group with 'baseName'
        // unless alread added.  The corresponding index in '_indices' will be
        // incremented if a new group is created.
        // 'eq_deps': pre-computed dependencies between equations.
        // Returns whether a new group was created.
        virtual bool addExprToGroup(EqualsExprPtr eq,
                                    BoolExprPtr cond, // may be nullptr.
                                    const string& baseName,
                                    EqDepMap& eq_deps);

    public:
        EqGroups() {}
        EqGroups(const string& basename_default, Dimensions& dims) :
            _basename_default(basename_default),
            _dims(&dims) {}
        virtual ~EqGroups() {}

        virtual void set_basename_default(const string& basename_default) {
            _basename_default = basename_default;
        }
        virtual void set_dims(Dimensions& dims) {
            _dims = &dims;
        }
        
        // Separate a set of equations into eqGroups based
        // on the target string.
        // Target string is a comma-separated list of key-value pairs, e.g.,
        // "eqGroup1=foo,eqGroup2=bar".
        // In this example, all eqs updating grid names containing 'foo' go in eqGroup1,
        // all eqs updating grid names containing 'bar' go in eqGroup2, and
        // each remaining eq goes into a separate eqGroup.
        void makeEqGroups(Eqs& eqs,
                          const string& targets,
                          EqDepMap& eq_deps,
                          std::ostream& os);
        void makeEqGroups(Eqs& eqs,
                          const string& targets,
                          IntTuple& pts,
                          bool find_deps,
                          std::ostream& os) {
            EqDepMap eq_deps;
            if (find_deps)
                eqs.findDeps(pts, *_dims, &eq_deps, os);
            makeEqGroups(eqs, targets, eq_deps, os);
        }

        virtual const Grids& getOutputGrids() const {
            return _outGrids;
        }

        // Visit all the equations in all eqGroups.
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

        // Reorder groups based on dependencies.
        virtual void sort();
    
        // Print a list of eqGroups.
        virtual void printInfo(ostream& os) const {
            os << "Identified stencil equation-groups:" << endl;
            for (auto& eq : *this) {
                for (auto gp : eq.getOutputGrids()) {
                    string eqName = eq.getName();
                    os << "  Equation group '" << eqName << "' updates grid '" <<
                        gp->getName() << "'." << endl;
                }
            }
        }

        // Print stats for the equation(s) in all groups.
        virtual void printStats(ostream& os, const string& msg);

        // Apply optimizations requested in settings.
        void optimizeEqGroups(CompilerSettings& settings,
                              const string& descr,
                              bool printSets,
                              ostream& os);
    };
    
} // namespace yask.
    
#endif
