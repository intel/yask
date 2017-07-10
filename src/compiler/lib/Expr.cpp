/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

///////// Stencil AST Expressions. ////////////

// TODO: break this up into several smaller files.

#include "Print.hpp"
#include "ExprUtils.hpp"
#include "Parse.hpp"
#include "Print.hpp"
#include "CppIntrin.hpp"

namespace yask {

    // Stencil-solution APIs.
    yc_grid_ptr StencilSolution::new_grid(const std::string& name,
                                          const std::vector<yc_index_node_ptr>& dims) {
        
        // Make new grid and add to solution.
        // FIXME: mem leak--delete this in dtor or make smart ptr.

        // Copy pointers to concrete type.
        IndexExprPtrVec dims2;
        for (auto d : dims) {
            auto d2 = dynamic_pointer_cast<IndexExpr>(d);
            assert(d2);
            dims2.push_back(d2);
        }
        
        auto* gp = new Grid(name, this, dims2);
        assert(gp);
        return gp;
    }

    // Stencil-solution APIs.
    yc_grid_ptr StencilSolution::new_grid(const std::string& name,
                                          const yc_index_node_ptr dim1,
                                          const yc_index_node_ptr dim2,
                                          const yc_index_node_ptr dim3,
                                          const yc_index_node_ptr dim4,
                                          const yc_index_node_ptr dim5,
                                          const yc_index_node_ptr dim6) {

        // Add dims that are not null ptrs.
        vector<yc_index_node_ptr> dims;
        if (dim1)
            dims.push_back(dim1);
        if (dim2)
            dims.push_back(dim2);
        if (dim3)
            dims.push_back(dim3);
        if (dim4)
            dims.push_back(dim4);
        if (dim5)
            dims.push_back(dim5);
        if (dim6)
            dims.push_back(dim6);

        return new_grid(name, dims);
    }

    void StencilSolution::set_fold_len(const yc_index_node_ptr dim,
                                       int len) {
        auto& fold = _settings._foldOptions;
        fold.addDimBack(dim->get_name(), len);
    }
    void StencilSolution::set_cluster_mult(const yc_index_node_ptr dim,
                                           int mult) {
        auto& cluster = _settings._clusterOptions;
        cluster.addDimBack(dim->get_name(), mult);
    }

    // Create the intermediate data for printing.
    void StencilSolution::analyze_solution(int vlen,
                                           bool is_folding_efficient,
                                           ostream& os) {

        // Find all the stencil dimensions from the grids.
        // Create the final folds and clusters from the cmd-line options.
        _dims.setDims(_grids, _settings, vlen, is_folding_efficient, os);

        // Call the stencil 'define' method to create ASTs.
        // ASTs can also be created via the APIs.
        define();

        // Check for illegal dependencies within equations for scalar size.
        if (_settings._find_deps) {
            os << "Checking equation(s) with scalar operations...\n"
                " If this fails, review stencil equation(s) for illegal dependencies.\n";
            _eqs.checkDeps(_dims._scalar, _dims._stepDim);
        }

        // Check for illegal dependencies within equations for vector size.
        if (_settings._find_deps) {
            os << "Checking equation(s) with folded-vector operations...\n"
                " If this fails, the fold dimensions are not compatible with all equations.\n";
            _eqs.checkDeps(_dims._fold, _dims._stepDim);
        }

        // Check for illegal dependencies within equations for cluster size and
        // also create equation groups based on legal dependencies.
        os << "Checking equation(s) with clusters of vectors...\n"
            " If this fails, the cluster dimensions are not compatible with all equations.\n";
        _eqGroups.set_basename_default(_settings._eq_group_basename_default);
        _eqGroups.set_dims(_dims);
        _eqGroups.makeEqGroups(_eqs, _settings._eqGroupTargets,
                               _dims._clusterPts, _settings._find_deps);
        _eqGroups.optimizeEqGroups(_settings, "scalar & vector", false, os);

        // Make a copy of each equation at each cluster offset.
        // We will use these for inter-cluster optimizations and code generation.
        os << "Constructing cluster of equations containing " <<
            _dims._clusterMults.product() << " vector(s)...\n";
        _clusterEqGroups = _eqGroups;
        _clusterEqGroups.replicateEqsInCluster(_dims);
        if (_settings._doOptCluster)
            _clusterEqGroups.optimizeEqGroups(_settings, "cluster", true, cout);
    }

    // Format in given format-type.
    string StencilSolution::format(const string& format_type,
                                   ostream& os) {
        // Look for format match.
        // Most args to the printers just set references to data.
        // Data itself will be created in analyze_solution().
        PrinterBase* printer = 0;
        if (format_type == "cpp")
            printer = new YASKCppPrinter(*this, _eqGroups, _clusterEqGroups, _dims);
        else if (format_type == "knc")
            printer = new YASKKncPrinter(*this, _eqGroups, _clusterEqGroups, _dims);
        else if (format_type == "avx" || format_type == "avx2")
            printer = new YASKAvx256Printer(*this, _eqGroups, _clusterEqGroups, _dims);
        else if (format_type == "avx512")
            printer = new YASKAvx512Printer(*this, _eqGroups, _clusterEqGroups, _dims);
        else if (format_type == "dot")
            printer = new DOTPrinter(*this, _clusterEqGroups, false);
        else if (format_type == "dot-lite")
            printer = new DOTPrinter(*this, _clusterEqGroups, true);
        else if (format_type == "pseudo")
            printer = new PseudoPrinter(*this, _clusterEqGroups);
        else if (format_type == "pov-ray") // undocumented.
            printer = new POVRayPrinter(*this, _clusterEqGroups);
        else {
            cerr << "Error: format-type '" << format_type <<
                "' is not recognized." << endl;
            exit(1);
        }
        assert(printer);
        int vlen = printer->num_vec_elems();
        bool is_folding_efficient = printer->is_folding_efficient();

        // Set data for equation groups, dims, etc.
        analyze_solution(vlen, is_folding_efficient, os);

        // Create the output.
        os << "Generating '" << format_type << "' output...\n";
        string res = printer->format();
        delete printer;

        return res;
    }
    void StencilSolution::write(const std::string& filename,
                                const std::string& format_type,
                                bool debug) {

        // Get file stream.
        ostream* os = 0;
        ofstream* ofs = 0;

        // Use '-' for stdout.
        if (filename == "-")
            os = &cout;
        else {
            ofs = new ofstream(filename, ofstream::out | ofstream::trunc);
            if (!ofs || !ofs->is_open()) {
                cerr << "Error: cannot open '" << filename <<
                    "' for output.\n";
                exit(1);
            }
            os = ofs;
        }
        assert(os);

        // Create output.
        string res = format(format_type, debug);

        // Send to stream.
        *os << res;

        // Close file if needed.
        if (ofs) {
            ofs->close();
            delete ofs;
        }
    }

    // grid APIs.
    yc_grid_point_node_ptr
    Grid::new_relative_grid_point(std::vector<int> dim_offsets) {

        // Check for correct number of indices.
        if (_dims.size() != dim_offsets.size()) {
            cerr << "Error: attempt to create a relative grid point in " <<
                _dims.size() << "D grid '" << _name << "' with " <<
                dim_offsets.size() << " indices.\n";
            exit(1);
        }

        // Check dim types.
        // Make default args w/just index.
        NumExprPtrVec args;
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            if (dim->getType() == MISC_INDEX) {
                cerr << "Error: attempt to create a relative grid point in " <<
                    _dims.size() << "D grid '" << _name <<
                    "' containing non-step or non-domain dim '" <<
                    dim->getName() << "'.\n";
                exit(1);
            }
            auto ie = dim->clone();
            args.push_back(ie);
        }
        
        // Create a point from the args.
        GridPointPtr gpp = make_shared<GridPoint>(this, args);

        // Modify the offsets.
        for (size_t i = 0; i < _dims.size(); i++) {
            auto dim = _dims.at(i);
            IntScalar ofs(dim->getName(), dim_offsets.at(i));
            gpp->setArgOffset(ofs);
        }
        return gpp;
    }

    yc_grid_point_node_ptr
    Grid::new_relative_grid_point(int dim1_offset,
                                  int dim2_offset,
                                  int dim3_offset,
                                  int dim4_offset,
                                  int dim5_offset,
                                  int dim6_offset) {
        std::vector<int> dim_offsets;
        auto n = _dims.size();
        if (n >= 1)
            dim_offsets.push_back(dim1_offset);
        if (n >= 2)
            dim_offsets.push_back(dim2_offset);
        if (n >= 3)
            dim_offsets.push_back(dim3_offset);
        if (n >= 4)
            dim_offsets.push_back(dim4_offset);
        if (n >= 5)
            dim_offsets.push_back(dim5_offset);
        if (n >= 6)
            dim_offsets.push_back(dim6_offset);
        if (n >= 7) {
            cerr << "Error: " << n << "-D grid not supported.\n";
            exit(1);
        }
        return new_relative_grid_point(dim_offsets);
    }
    vector<string> Grid::get_dim_names() const {
        vector<string> ret;
        for (auto dn : getDims())
            ret.push_back(dn->getName());
        return ret;
    }

    // grid_point APIs.
    yc_grid* GridPoint::get_grid() {
        return _grid;
    }
    
    // yask_compiler_factory API methods.
    yc_solution_ptr
    yc_factory::new_solution(const std::string& name) const {
        return make_shared<EmptyStencil>(name);
    }
    
    //node_factory API methods.
    yc_equation_node_ptr
    yc_node_factory::new_equation_node(yc_grid_point_node_ptr lhs,
                                       yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<GridPoint>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return operator EQUALS_OPER(lp, rp);
    }
    yc_const_number_node_ptr
    yc_node_factory::new_const_number_node(double val) {
        return make_shared<ConstExpr>(val);
    }
    yc_negate_node_ptr
    yc_node_factory::new_negate_node(yc_number_node_ptr rhs) {
        auto p = dynamic_pointer_cast<NumExpr>(rhs);
        assert(p);
        return make_shared<NegExpr>(p);
    }
    yc_add_node_ptr
    yc_node_factory::new_add_node(yc_number_node_ptr lhs,
                                  yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<AddExpr>(lp, rp);
    }
    yc_multiply_node_ptr
    yc_node_factory::new_multiply_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<MultExpr>(lp, rp);
    }
    yc_subtract_node_ptr
    yc_node_factory::new_subtract_node(yc_number_node_ptr lhs,
                                       yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<SubExpr>(lp, rp);
    }
    yc_divide_node_ptr
    yc_node_factory::new_divide_node(yc_number_node_ptr lhs,
                                     yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<DivExpr>(lp, rp);
    }

    // Compare 2 expr pointers and return whether the expressions are
    // equivalent.
    // TODO: Be much smarter about matching symbolically-equivalent exprs.
    bool areExprsSame(const Expr* e1, const Expr* e2) {

        // Handle null pointers.
        if (e1 == NULL && e2 == NULL)
            return true;
        if (e1 == NULL || e2 == NULL)
            return false;

        // Neither are null, so compare contents.
        return e1->isSame(e2);
    }

    // Unary.
    NumExprPtr operator-(const NumExprPtr rhs) {
        return make_shared<NegExpr>(rhs);
    }

    // A free function to create a constant expression.
    NumExprPtr constNum(double rhs) {
        return make_shared<ConstExpr>(rhs);
    }

    // Free functions to create boundary indices, e.g., 'first_index(x)'.
    NumExprPtr first_index(IndexExprPtr dim) {
        assert(dim->getType() == DOMAIN_INDEX);
        return make_shared<IndexExpr>(dim->getName(), FIRST_INDEX);
    }
    NumExprPtr last_index(IndexExprPtr dim) {
        assert(dim->getType() == DOMAIN_INDEX);
        return make_shared<IndexExpr>(dim->getName(), LAST_INDEX);
    }
    
    // Commutative.
    // If one side is nothing, just return other side;
    // This allows us to start with an uninitialized GridValue
    // and do the right thing.
    // Start with an empty expression.
    // If LHS is the same expr type, add its operands.
    // Othewise, add the whole expr.
    // Repeat for RHS.
#define COMM_OPER(oper, exprtype)                       \
    NumExprPtr operator oper(const NumExprPtr lhs,      \
                             const NumExprPtr rhs) {    \
        if (lhs.get() == NULL)                          \
            return rhs;                                 \
        else if (rhs.get() == NULL)                     \
            return lhs;                                 \
        auto ex = make_shared<exprtype>();              \
        ex->mergeExpr(lhs);                             \
        ex->mergeExpr(rhs);                             \
        return ex;                                      \
    }                                                   \
    NumExprPtr operator oper(double lhs,                \
                             const NumExprPtr rhs) {    \
        NumExprPtr p = make_shared<ConstExpr>(lhs);     \
        return p oper rhs;                              \
    }                                                   \
    NumExprPtr operator oper(const NumExprPtr lhs,      \
                             double rhs) {              \
        NumExprPtr p = make_shared<ConstExpr>(rhs);     \
        return lhs oper p;                              \
    }
    COMM_OPER(+, AddExpr)
    COMM_OPER(*, MultExpr)

    // Self-modifying versions.
    void operator+=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs + rhs;
    }
    void operator+=(NumExprPtr& lhs, double rhs) {
        lhs = lhs + rhs;
    }
    void operator*=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs * rhs;
    }
    void operator*=(NumExprPtr& lhs, double rhs) {
        lhs = lhs * rhs;
    }

    // Binary.
    NumExprPtr operator-(const NumExprPtr lhs, const NumExprPtr rhs) {
#ifdef USE_ADD_NEG
        // Generate A + -B instead of A - B to allow easy reordering.
        NumExprPtr nrhs = make_shared<NegExpr>(rhs);
        return lhs + nrhs;
#else
        return make_shared<SubExpr>(lhs, rhs);
#endif    
    }
    NumExprPtr operator-(double lhs, const NumExprPtr rhs) {
        NumExprPtr p = make_shared<ConstExpr>(lhs);
        return p - rhs;
    }
    NumExprPtr operator-(const NumExprPtr lhs, double rhs) {
        NumExprPtr p = make_shared<ConstExpr>(rhs);
        return lhs - p;
    }

    void operator-=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs - rhs;
    }
    void operator-=(NumExprPtr& lhs, double rhs) {
        lhs = lhs - rhs;
    }

    NumExprPtr operator/(const NumExprPtr lhs, const NumExprPtr rhs) {
        return make_shared<DivExpr>(lhs, rhs);
    }
    NumExprPtr operator/(double lhs, const NumExprPtr rhs) {
        NumExprPtr p = make_shared<ConstExpr>(lhs);
        return p / rhs;
    }
    NumExprPtr operator/(const NumExprPtr lhs, double rhs) {
        NumExprPtr p = make_shared<ConstExpr>(rhs);
        return lhs / p;
    }

    void operator/=(NumExprPtr& lhs, const NumExprPtr rhs) {
        lhs = lhs / rhs;
    }
    void operator/=(NumExprPtr& lhs, double rhs) {
        lhs = lhs / rhs;
    }

    // Define a conditional.
    IfExprPtr operator IF_OPER(EqualsExprPtr expr, const BoolExprPtr cond) {

        // Get to list of equations.
        auto gpp = expr->getLhs();
        assert(gpp);
        Grid* gp = gpp->getGrid();
        assert(gp);
        auto* soln = gp->getSoln();
        assert(soln);
        auto& eqs = soln->getEqs();
    
        // Make if-expression node.
        auto ifp = make_shared<IfExpr>(expr, cond);

        // Save expr and if-cond.
        eqs.addCondEq(expr, cond);

        return ifp;
    }

    // Define the value of a grid point.
    // Add this equation to the list of eqs for this stencil.
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, const NumExprPtr rhs) {

        // Get to list of equations.
        Grid* gp = gpp->getGrid();
        assert(gp);
        auto* soln = gp->getSoln();
        assert(soln);
        auto& eqs = soln->getEqs();
    
        // TODO: check validity of LHS (gpp).
        
        // Make expression node.
        auto expr = make_shared<EqualsExpr>(gpp, rhs);

        // Save the expression.
        eqs.addEq(expr);

        return expr;
    }
    EqualsExprPtr operator EQUALS_OPER(GridPointPtr gpp, double rhs) {
        return gpp EQUALS_OPER constNum(rhs);
    }

    // Visitor acceptors.
    void ConstExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void CodeExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryNumExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryBoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void UnaryNum2BoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryNumExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryBoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    template<>
    void BinaryNum2BoolExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void CommutativeExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void GridPoint::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void EqualsExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void IfExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }
    void IndexExpr::accept(ExprVisitor* ev) {
        ev->visit(this);
    }

    // EqualsExpr methods.
    bool EqualsExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const EqualsExpr*>(other);
        return p && _opStr == p->_opStr &&
            _lhs->isSame(p->_lhs.get()) &&
            _rhs->isSame(p->_rhs.get());
    }

    // IfExpr methods.
    bool IfExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const IfExpr*>(other);
        return p &&
            areExprsSame(_expr, p->_expr) &&
            areExprsSame(_rhs, p->_rhs);
    }

    // Commutative methods.
    bool CommutativeExpr::isSame(const Expr* other) const {
        auto p = dynamic_cast<const CommutativeExpr*>(other);
        if (!p || _opStr != p->_opStr)
            return false;
        if (_ops.size() != p->_ops.size())
            return false;
        
        // Operands must be the same, but not necessarily in same order.  This
        // tracks the indices in 'other' that have already been matched.
        set<size_t> matches;

        // Loop through this set of ops.
        for (auto op : _ops) {

            // Loop through other set of ops, looking for match.
            bool found = false;
            for (size_t i = 0; i < p->_ops.size(); i++) {
                auto oop = p->_ops[i];

                // check unless already matched.
                if (matches.count(i) == 0 && op->isSame(oop.get())) {
                    matches.insert(i);
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }

        // Do all match?
        return matches.size() == _ops.size();
    }

    // GridPoint methods.
    GridPoint::GridPoint(Grid* grid, const NumExprPtrVec& args) :
        _grid(grid), _args(args) {

        // Check for correct number of args.
        size_t nd = grid->getDims().size();
        if (nd != args.size()) {
            cerr << "Error: attempt to create a grid point in " <<
                nd << "-D grid '" << getName() << "' with " <<
                args.size() << " indices.\n";
            exit(1);
        }

        // Eval each arg.
#ifdef DEBUG_GP
        cout << "Creating grid point " << makeQuotedStr() << "...\n";
#endif
        auto dims = grid->getDims();
        for (size_t i = 0; i < nd; i++) {
            auto dim = dims.at(i);
            auto dname = dim->getName();
            auto arg = args.at(i);
#ifdef DEBUG_GP
            cout << " Arg " << arg->makeQuotedStr() <<
                " at dim '" << dname << "'\n";
#endif
            int offset = 0;

            // A compile-time const?
            if (arg->isConstVal()) {
                _consts.addDimBack(dname, arg->getIntVal());
#ifdef DEBUG_GP
                cout << "  is const val " << arg->getIntVal() << endl;
#endif
            }

            // A simple offset?
            else if (arg->isOffsetFrom(dname, offset)) {
                _offsets.addDimBack(dname, offset);
#ifdef DEBUG_GP
                cout << "  has offset " << offset << endl;
#endif
            }
        }
    }
    const string& GridPoint::getName() const {
        return _grid->getName();
    }
    bool GridPoint::operator==(const GridPoint& rhs) const {
        return makeStr() == rhs.makeStr(); // TODO: make more efficient.
    }
    bool GridPoint::operator<(const GridPoint& rhs) const {
        return makeStr() < rhs.makeStr(); // TODO: make more efficient.
    }
    bool GridPoint::isAheadOfInDir(const GridPoint& rhs, const IntScalar& dir) const {
        return _grid == rhs._grid && // must be same var.
            _offsets.areDimsSame(rhs._offsets) &&
            _offsets.isAheadOfInDir(rhs._offsets, dir);
    }
    string GridPoint::makeArgsStr() const {
        string str;
        int i = 0;
        for (auto arg : _args) {
            if (i++) str += ", ";
            str += arg->makeStr();
        }
        return str;
    }
    string GridPoint::makeStr() const {
        string str = _grid->getName() + "(" +
                             makeArgsStr() + ")";
        return str;
    }

    // Set given arg to given offset; ignore if not in step or domain grid dims.
    void GridPoint::setArgOffset(const IntScalar& offset) {

        // Find dim in grid.
        auto gdims = _grid->getDims();
        for (size_t i = 0; i < gdims.size(); i++) {
            auto gdim = gdims[i];

            // Must be domain or step dim.
            if (gdim->getType() == MISC_INDEX)
                continue;
            
            auto dname = gdim->getName();
            if (offset.getName() == dname) {

                // Make offset equation.
                int ofs = offset.getVal();
                auto ie = gdim->clone();
                NumExprPtr nep;
                if (ofs > 0) {
                    auto op = constNum(ofs);
                    nep = make_shared<AddExpr>(ie, op);
                }
                else if (ofs < 0) {
                    auto op = constNum(-ofs);
                    nep = make_shared<SubExpr>(ie, op);
                }
                else                // 0 offset.
                    nep = ie;

                // Replace in args.
                _args[i] = nep;

                // Set offset.
                _offsets.addDimBack(dname, ofs);

                // Remove const.
                _consts = _consts.removeDim(dname);

                break;
            }
        }
    }
    
    // Make string like "x+(4/VLEN_X), y, z-(2/VLEN_Z)" from
    // original args "x+4, y, z-2".
    // This object has numerators; norm object has denominators.
    // Args w/o simple offset are not modified.
    string GridPoint::makeNormArgsStr(const IntTuple& fold) const {

        ostringstream oss;
        auto gd = getGrid()->getDims();
        for (size_t i = 0; i < gd.size(); i++) {
            if (i)
                oss << ", ";
            auto dname = gd[i]->getName();
            
            // Non-0 offset and exists in fold?
            auto* ofs = _offsets.lookup(dname);
            if (ofs && *ofs && fold.lookup(dname)) {
                oss << "(" << dname;

                // Positive offset, e.g., 'xv + (4 / VLEN_X)'.
                if (*ofs > 0)
                    oss << " + (" << *ofs;

                // Neg offset, e.g., 'xv - (4 / VLEN_X)'.
                // Put '-' sign outside division to fix truncated division problem.
                else
                    oss << " - (" << (- *ofs);
                    
                // add divisor.
                string cap_dname = dname;
                transform(cap_dname.begin(), cap_dname.end(), cap_dname.begin(), ::toupper);
                oss << " / VLEN_" << cap_dname << "))";
            }

            // Otherise, just use given arg.
            else
                oss << _args.at(i)->makeStr();
        }
        return oss.str();
    }
    
    // Is this expr a simple offset?
    bool IndexExpr::isOffsetFrom(string dim, int& offset) {

        // An index expr is an offset if it's a step or domain dim and the
        // dims are the same.
        if (_type != MISC_INDEX && _dimName == dim) {
            offset = 0;
            return true;
        }
        return false;
    }
    bool DivExpr::isOffsetFrom(string dim, int& offset) {

        // Could allow 'dim / 1', but seems silly.
        return false;
    }
    bool MultExpr::isOffsetFrom(string dim, int& offset) {

        // Could allow 'dim * 1', but seems silly.
        return false;
    }
    bool SubExpr::isOffsetFrom(string dim, int& offset) {

        // Is this of the form 'dim - offset'?
        int tmp = 0;
        if (_lhs->isOffsetFrom(dim, tmp) &&
            _rhs->isConstVal()) {
            offset = tmp - _rhs->getIntVal();
            return true;
        }
        return false;
    }
    bool AddExpr::isOffsetFrom(string dim, int& offset) {

        // Is this of the form 'dim + offset'?
        // Allow any similar form, e.g., '-5 + dim + 2'.
        int sum = 0;
        int num_dims = 0;
        int tmp = 0;
        for (auto op : _ops) {

            // Is this operand 'dim'?
            if (op->isOffsetFrom(dim, tmp))
                num_dims++;

            // Is this operand a const int?
            else if (op->isConstVal())
                sum += op->getIntVal();

            // Anything else isn't allowed.
            else
                return false;
        }
        // Must be exactly one 'dim'.
        // Don't allow silly forms like 'dim - dim + dim + 2'.
        if (num_dims == 1) {
            offset = tmp + sum;
            return true;
        }
        return false;
    }
    
    
    // Make a readable string from an expression.
    string Expr::makeStr() const {
        ostringstream oss;
    
        // Use a print visitor to make a string.
        PrintHelper ph(NULL, "temp", "", "", "");
        PrintVisitorTopDown pv(oss, ph);
        accept(&pv);

        // Return anything written to the stream
        // concatenated with anything left in the
        // PrintVisitor.
        return oss.str() + pv.getExprStr();
    }

    // Return number of nodes.
    int Expr::getNumNodes() const {

        // Use a counter visitor.
        CounterVisitor cv;
        accept(&cv);

        return cv.getNumNodes();
    }

    // Const version of accept.
    void Expr::accept(ExprVisitor* ev) const {
        const_cast<Expr*>(this)->accept(ev);
    }

    // Create an expression to a specific point in this grid.
    // Note that this doesn't actually 'read' or 'write' a value;
    // it's just a node in an expression.
    GridPointPtr Grid::makePoint(const NumExprPtrVec& args) {
        auto gpp = make_shared<GridPoint>(this, args);
        return gpp;
    }

    // Ctor for Grid.
    Grid::Grid(string name, StencilSolution* soln,
               IndexExprPtr dim1,
               IndexExprPtr dim2,
               IndexExprPtr dim3,
               IndexExprPtr dim4,
               IndexExprPtr dim5,
               IndexExprPtr dim6) :
            _name(name),       // TODO: validate that name is legal C++ var.
            _soln(soln) {

        // Register in soln.
        if (soln)
            soln->getGrids().insert(this);

        // Add dims that are not null.
        if (dim1)
            _dims.push_back(dim1);
        if (dim2)
            _dims.push_back(dim2);
        if (dim3)
            _dims.push_back(dim3);
        if (dim4)
            _dims.push_back(dim4);
        if (dim5)
            _dims.push_back(dim5);
        if (dim6)
            _dims.push_back(dim6);
    }
    Grid::Grid(string name, StencilSolution* soln,
               const IndexExprPtrVec& dims) :
        Grid(name, soln) {
        _dims = dims;
    }
    
    // Update halos based on each value in 'offsets' in some
    // read or write to this grid.
    void Grid::updateHalo(const IntTuple& offsets)
    {
        // Find step value or use 0 if none.
        int stepVal = 0;
        auto stepDim = getStepDim();
        if (stepDim) {
            auto* p = offsets.lookup(stepDim->getName());
            if (p)
                stepVal = *p;
        }
        auto& halos = _halos[stepVal];

        // Update halo vals.
        for (auto& dim : offsets.getDims()) {
            auto& dname = dim.getName();
            int val = abs(dim.getVal());

            // Don't keep halo in step dim.
            if (stepDim && dname == stepDim->getName())
                continue;

            auto* p = halos.lookup(dname);
            if (!p)
                halos.addDimBack(dname, val);
            else if (val > *p)
                *p = val;
            // else, current value is larger than val, so don't update.
        }
    }

    // Update const indices based on 'indices'.
    void Grid::updateConstIndices(const IntTuple& indices) {

        for (auto& dim : indices.getDims()) {
            auto& dname = dim.getName();
            int val = dim.getVal();

            // Update min.
            auto* minp = _minIndices.lookup(dname);
            if (!minp)
                _minIndices.addDimBack(dname, val);
            else if (val < *minp)
                *minp = val;

            // Update max.
            auto* maxp = _maxIndices.lookup(dname);
            if (!maxp)
                _maxIndices.addDimBack(dname, val);
            else if (val > *maxp)
                *maxp = val;
        }
    }

    // Determine how many values in step-dim are needed.
    int Grid::getStepDimSize() const
    {
        // Only need one value if no step-dim index used.
        auto stepDim = getStepDim();
        if (!stepDim)
            return 1;

        // No info stored?
        if (_halos.size() == 0)
            return 1;

        // Find halos at min and max step-dim points.
        // These should correspond to the 1st read and only write points.
        auto first_i = _halos.cbegin(); // begin == first.
        auto last_i = _halos.crbegin(); // reverse-begin == last.
        int first_ofs = first_i->first;
        int last_ofs = last_i->first;
        auto& first_halo = first_i->second;
        auto& last_halo = last_i->second;

        // Default step-dim size is range of offsets.
        assert(last_ofs >= first_ofs);
        int sz = last_ofs - first_ofs + 1;
    
        // If first and last halos are zero, we can further optimize storage by
        // immediately reusing memory location.
        if (sz > 1 &&
            first_halo.max() == 0 &&
            last_halo.max() == 0)
            sz--;

        // TODO: recognize that reading in one equation and then writing in
        // another can also reuse storage.

        return sz;
    }

    // A visitor to collect grids and points visited in a set of eqs.
    class PointVisitor : public ExprVisitor {

        // A set of all points to ensure pointers to each
        // unique point have same value.
        set<GridPoint> _all_pts;

        // A type to hold a mapping of equations to a set of grids in each.
        typedef unordered_set<Grid*> GridSet;
        typedef unordered_map<EqualsExpr*, GridSet> GridMap;

        GridMap _lhs_grids; // outputs of eqs.
        GridMap _rhs_grids; // inputs of eqs.
    
        // A type to hold a mapping of equations to a set of points in each.
        typedef unordered_set<const GridPoint*> PointSet;
        typedef unordered_map<EqualsExpr*, PointSet> PointMap;

        PointMap _lhs_pts; // outputs of eqs.
        PointMap _rhs_pts; // inputs of eqs.

        IntTuple* _pts=0;    // Points to visit from each index.
        EqualsExpr* _eq=0;   // Current equation.

        // Add a grid point to _all_pts and get a pointer to it.
        // If matching point exists, just get a pointer to existing one.
        const GridPoint* _add_pt(const GridPoint& gpt0, const IntTuple& offsets) {
            auto gpt1 = gpt0;
            gpt1.setArgOffsets(offsets);
            auto i = _all_pts.insert(gpt1);
            auto& gpt2 = *i.first;
            return &gpt2;
        }

        // Add all points with _pts offsets from gpt0 to 'pt_map'.
        // Add grid from gpt0 to 'grid_map'.
        // Maps keyed by _eq.
        void _add_pts(GridPoint* gpt0, GridMap& grid_map, PointMap& pt_map) {

            // Add grid.
            auto* g = gpt0->getGrid();
            grid_map[_eq].insert(g);

            // Visit each point in _pts.
            if (_pts) {
                _pts->visitAllPoints([&](const IntTuple& ptofs){

                        // Add offset to gpt0.
                        auto& pt0 = gpt0->getArgOffsets();
                        auto pt1 = pt0.addElements(ptofs, false);
                        auto* p = _add_pt(*gpt0, pt1);
                        pt_map[_eq].insert(p);
                    });
            }

            // Visit one point.
            else {
                pt_map[_eq].insert(gpt0);
            }
        }   
    
    public:
    
        // Ctor.
        // 'pts' contains offsets from each point to create.
        PointVisitor() :
            _pts(0) {}
        PointVisitor(IntTuple& pts) :
            _pts(&pts) {}
        virtual ~PointVisitor() {}

        GridMap& getOutputGrids() { return _lhs_grids; }
        GridMap& getInputGrids() { return _rhs_grids; }
        PointMap& getOutputPts() { return _lhs_pts; }
        PointMap& getInputPts() { return _rhs_pts; }
        int getNumEqs() const { return (int)_lhs_pts.size(); }

        // Determine whether 2 sets have any common points.
        virtual bool do_sets_intersect(const GridSet& a,
                                       const GridSet& b) {
            for (auto ai : a) {
                if (b.count(ai) > 0)
                    return true;
            }
            return false;
        }
        virtual bool do_sets_intersect(const PointSet& a,
                                       const PointSet& b) {
            for (auto ai : a) {
                if (b.count(ai) > 0)
                    return true;
            }
            return false;
        }
    
        // Callback at an equality.
        virtual void visit(EqualsExpr* ee) {

            // Remember this equation.
            _eq = ee;

            // Make sure map entries exist.
            _lhs_grids[_eq];
            _rhs_grids[_eq];
            _lhs_pts[_eq];
            _rhs_pts[_eq];

            // Store all LHS points.
            auto* lhs = ee->getLhs().get();
            _add_pts(lhs, _lhs_grids, _lhs_pts);

            // visit RHS.
            NumExprPtr rhs = ee->getRhs();
            rhs->accept(this);

            // Don't visit LHS because we've already saved it.
        }

        // Callback at a grid point on the RHS.
        virtual void visit(GridPoint* gp) {
            assert(_eq);

            // Store all RHS points.
            _add_pts(gp, _rhs_grids, _rhs_pts);
        }
    };

    // Recursive search starting at 'a'.
    // Fill in _full_deps.
    bool EqDeps::_analyze(EqualsExprPtr a, SeenSet* seen)
    {
        // 'a' already visited?
        bool was_seen = (seen && seen->count(a));
        if (was_seen)
            return true;
    
        // any dependencies?
        if (_deps.count(a)) {
            auto& adeps = _deps.at(a);
    
            // make new seen-set adding 'a'.
            SeenSet seen1;
            if (seen)
                seen1 = *seen; // copy nodes already seen.
            seen1.insert(a);   // add this one.
        
            // loop thru dependences 'a' -> 'b'.
            for (auto b : adeps) {

                // whole path up to and including 'a' depends on 'b'.
                for (auto p : seen1)
                    _full_deps[p].insert(b);
            
                // follow path.
                _analyze(b, &seen1);
            }
        }

        // no dependence; make an empty entry.
        return false;
    }

    // Find dependencies based on all eqs.
    // If 'eq_deps' is set, save dependencies between eqs.
    // [BIG]TODO: replace dependency algorithms with integration of a polyhedral
    // library.
    void Eqs::findDeps(IntTuple& pts,
                       const string& stepDim,
                       EqDepMap* eq_deps) {

        // Gather points from all eqs in all grids.
        PointVisitor pt_vis(pts);

        // Gather initial stats from all eqs.
        cout << " Scanning " << getEqs().size() << " equations(s)...\n";
        for (auto eq1 : getEqs())
            eq1->accept(&pt_vis);
        auto& outGrids = pt_vis.getOutputGrids();
        auto& inGrids = pt_vis.getInputGrids();
        auto& outPts = pt_vis.getOutputPts();
        auto& inPts = pt_vis.getInputPts();
        
        // Check dependencies on all eqs.
        for (auto eq1 : getEqs()) {
            auto* eq1p = eq1.get();
            assert(outGrids.count(eq1p));
            assert(inGrids.count(eq1p));
            auto& og1 = outGrids.at(eq1p);
            //auto& ig1 = inGrids.at(eq1p);
            auto& op1 = outPts.at(eq1p);
            auto& ip1 = inPts.at(eq1p);
            auto cond1 = getCond(eq1p);

            // An equation must update one grid only.
            assert(og1.size() == 1);
            auto* g1 = eq1->getGrid();
            assert(og1.count(g1));

            // Scan output (LHS) points.
            int si1 = 0;        // step index offset for LHS of eq1.
            for (auto i1 : op1) {
            
                // LHS of an equation must use step index.
                auto* si1p = i1->getArgOffsets().lookup(stepDim);
                if (!si1p) {
                    cerr << "Error: equation " << eq1->makeQuotedStr() <<
                        " does not use simple offset from step-dimension index var '" <<
                        stepDim << "' on LHS.\n";
                    exit(1);
                }
                assert(si1p);
                si1 = *si1p;
            }

            // Scan input (RHS) points.
            for (auto i1 : ip1) {

                // Check RHS of an equation that uses step index.
                auto* rsi1p = i1->getArgOffsets().lookup(stepDim);
                if (rsi1p) {
                    int rsi1 = *rsi1p;

                    // Cannot depend on future value in this dim.
                    if (rsi1 > si1) {
                        cerr << "Error: equation " << eq1->makeQuotedStr() <<
                            " contains an illegal dependence from offset " << rsi1 <<
                            " to " << si1 << " relative to step-dimension index var '" <<
                            stepDim << "'.\n";
                        exit(1);
                    }

                    // TODO: should make some dependency checks when rsi1 == si1.
                }
            }

            // TODO: check to make sure cond1 doesn't depend on stepDim.
            
#ifdef DEBUG_DEP
            cout << " Checking dependencies *within* equation " <<
                eq1->makeQuotedStr() << "...\n";
#endif

            // Find other eqs that depend on eq1.
            for (auto eq2 : getEqs()) {
                auto* eq2p = eq2.get();
                //auto& og2 = outGrids.at(eq2p);
                auto& ig2 = inGrids.at(eq2p);
                auto& op2 = outPts.at(eq2p);
                auto& ip2 = inPts.at(eq2p);
                auto cond2 = getCond(eq2p);

                bool same_eq = eq1 == eq2;
                bool same_cond = areExprsSame(cond1, cond2);

                // If two different eqs have the same condition, they
                // cannot update the exact same point.
                if (!same_eq && same_cond &&
                    pt_vis.do_sets_intersect(op1, op2)) {
                    cerr << "Error: two equations with condition " <<
                        cond1->makeQuotedStr() << " update the same point: " <<
                        eq1->makeQuotedStr() << " and " <<
                        eq2->makeQuotedStr() << endl;
                    exit(1);
                }

                // eq2 dep on eq1 => some output of eq1 is an input to eq2.
                // If the two eqs have the same condition, detect certain
                // dependencies by looking for exact matches.
                if (same_cond &&
                    pt_vis.do_sets_intersect(op1, ip2)) {

                    // Eq depends on itself?
                    if (same_eq) {
                                    
                        // Exit with error.
                        cerr << "Error: illegal dependency between LHS and RHS of equation " <<
                            eq1->makeQuotedStr() <<
                            " within offsets in range " << pts.makeDimValStr(" * ") << ".\n";
                        exit(1);
                    }

                    // Save dependency.
                    // Flag as both certain and possible because we need to follow
                    // certain ones when resolving indirect possible ones.
                    if (eq_deps) {
                        (*eq_deps)[certain_dep].set_dep_on(eq2, eq1);
                        (*eq_deps)[possible_dep].set_dep_on(eq2, eq1);
                    }
                        
                    // Move along to next eq2.
                    continue;
                }

                // Check more only if saving dependencies.
                if (!eq_deps)
                    continue;

                // Only check between different equations.
                if (same_eq)
                    continue;

                // Does eq1 define *any* point in a grid that eq2 inputs
                // at the same step index?  If so, they *might* have a
                // dependency. Some of these may not be real
                // dependencies due to conditions. Those that are real
                // may or may not be legal.
                //
                // Example:
                //  eq1: a(t+1, x, ...) EQUALS ... IF ... 
                //  eq2: b(t+1, x, ...) EQUALS a(t+1, x+5, ...) ... IF ...
                //
                // TODO: be much smarter about this and find only real
                // dependencies--use a polyhedral library?
                if (pt_vis.do_sets_intersect(og1, ig2)) {

                    // detailed check of g1 input points from eq2.
                    for (auto* i2 : ip2) {
                        if (i2->getGrid() != g1) continue;

                        // From same step index, e.g., same time?
                        auto* si2p = i2->getArgOffsets().lookup(stepDim);
                        if (si2p && (*si2p == si1)) {

                            // Save dependency.
                            if (eq_deps)
                                (*eq_deps)[possible_dep].set_dep_on(eq2, eq1);
                                
                            // Move along to next equation.
                            break;
                        }
                    }
                }
            }
        }

        // Resolve indirect dependencies.
        if (eq_deps) {
            cout << "  Resolving indirect dependencies...\n";
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1))
                (*eq_deps)[dt].analyze();
        }
        cout << " Done.\n";
    }

    // Get the full name of an eq-group.
    // Must be unique.
    string EqGroup::getName() const {

#if 0
        // Just use base name if zero index.
        if (!index)
            return baseName;
#endif

        // Add index to base name.
        ostringstream oss;
        oss << baseName << "_" << index;
        return oss.str();
    }

    // Make a description.
    string EqGroup::getDescription(bool show_cond,
                                   string quote) const
    {
        string des = "equation-group " + quote + getName() + quote;
        if (show_cond) {
            if (cond.get())
                des += " w/condition " + cond->makeQuotedStr(quote);
            else
                des += " w/no condition";
        }
        return des;
    }

    // Add an equation to an EqGroup
    // If 'update_stats', update grid and halo data.
    void EqGroup::addEq(EqualsExprPtr ee, bool update_stats)
    {
#ifdef DEBUG_EQ_GROUP
        cout << "EqGroup: adding " << ee->makeQuotedStr() << endl;
#endif
        _eqs.insert(ee);

        if (update_stats) {
    
            // Get I/O point data.
            PointVisitor pv;
            ee->accept(&pv);

            // update list of input and output grids.
            auto& outGrids = pv.getOutputGrids().at(ee.get());
            for (auto* g : outGrids)
                _outGrids.insert(g);
            auto& inGrids = pv.getInputGrids().at(ee.get());
            for (auto* g : inGrids)
                _inGrids.insert(g);

            // update halo info in grids.
            auto& outPts = pv.getOutputPts().at(ee.get());
            auto& inPts = pv.getInputPts().at(ee.get());
            auto& stepDim = _dims->_stepDim;

            // Output points.
            for (auto* op : outPts) {
                auto* g = op->getGrid();
                auto* g2 = const_cast<Grid*>(g); // need to update grid.
                g2->updateHalo(op->getArgOffsets());
                g2->updateConstIndices(op->getArgConsts());
            }
    
            // Input points.
            for (auto* ip : inPts) {
                auto* g = ip->getGrid();
                auto* g2 = const_cast<Grid*>(g); // need to update grid.
                g2->updateHalo(ip->getArgOffsets());
                g2->updateConstIndices(ip->getArgConsts());
            }
        }
    }
    
    // Set dependency on eg2 if this eq-group is dependent on it.
    // Return whether dependent.
    bool EqGroup::setDepOn(DepType dt, EqDepMap& eq_deps, const EqGroup& eg2)
    {

        // Eqs in this.
        for (auto& eq1 : getEqs()) {

            // Eqs in eg2.
            for (auto& eq2 : eg2.getEqs()) {

                if (eq_deps[dt].is_dep_on(eq1, eq2)) {

                    _dep_on[dt].insert(eg2.getName());
                    return true;
                }
            }
        }
        return false;
    }


    // Print stats from eqGroup.
    void EqGroup::printStats(ostream& os, const string& msg)
    {
        CounterVisitor cv;
        visitEqs(&cv);
        cv.printStats(os, msg);
    }

    // Visitor that will shift each grid point by an offset.
    class OffsetVisitor: public ExprVisitor {
        IntTuple _ofs;
    
    public:
        OffsetVisitor(const IntTuple& ofs) :
            _ofs(ofs) {}

        // Visit a grid point.
        virtual void visit(GridPoint* gp) {

            // Shift grid _ofs points.
            auto ofs0 = gp->getArgOffsets();
            IntTuple new_loc = ofs0.addElements(_ofs, false);
            gp->setArgOffsets(new_loc);
        }
    };

    // Replicate each equation at the non-zero offsets for
    // each vector in a cluster.
    void EqGroup::replicateEqsInCluster(Dimensions& dims)
    {
        // Make a copy of the original equations so we can iterate through
        // them while adding to the group.
        EqList eqs(_eqs);

        // Loop thru points in cluster.
        dims._clusterMults.visitAllPoints([&](const IntTuple& clusterIndex) {

                // Don't need copy of one at origin.
                if (clusterIndex.sum() > 0) {
            
                    // Get offset of cluster, which is each cluster index multipled
                    // by corresponding vector size.  Example: for a 4x4 fold in a
                    // 1x2 cluster, the 2nd cluster index will be (0,1) and the
                    // corresponding cluster offset will be (0,4).
                    auto clusterOffset = clusterIndex.multElements(dims._fold);

                    // Loop thru eqs.
                    for (auto eq : eqs) {
                        assert(eq.get());
            
                        // Make a copy.
                        auto eq2 = eq->cloneEquals();

                        // Add offsets to each grid point.
                        OffsetVisitor ov(clusterOffset);
                        eq2->accept(&ov);

                        // Put new equation into group.
                        addEq(eq2, false);
                    }
                }
            });

        // Ensure the expected number of equations now exist.
        assert(_eqs.size() == eqs.size() * dims._clusterMults.product());
    }

    // Reorder groups based on dependencies.
    void EqGroups::sort()
    {
        if (size() < 2)
            return;

        cout << " Sorting " << size() << " eq-group(s)...\n";

        // Want to keep original order as much as possible.
        // Only reorder if dependencies are in conflict.

        // Scan from beginning to end.
        for (size_t i = 0; i < size(); i++) {

            bool done = false;
            while (!done) {
        
                // Does eq-group[i] depend on any eq-group after it?
                auto& egi = at(i);
                for (size_t j = i+1; j < size(); j++) {
                
                    auto& egj = at(j);
                    bool do_swap = false;

                    // Must swap on certain deps.
                    if (egi.isDepOn(certain_dep, egj)) {
                        if (egj.isDepOn(certain_dep, egi)) {
                            cerr << "Error: circular dependency between eq-groups " <<
                                egi.getDescription() << " and " <<
                                egj.getDescription() << endl;
                            exit(1);
                        }
                        do_swap = true;
                    }

                    // Swap on possible deps if one-way.
                    else if (egi.isDepOn(possible_dep, egj) && 
                             !egj.isDepOn(possible_dep, egi) &&
                             !egj.isDepOn(certain_dep, egi)) {
                        do_swap = true;
                    }

                    if (do_swap) {

                        // Swap them.
                        EqGroup temp(egi);
                        egi = egj;
                        egj = temp;

                        // Start over at index i.
                        done = false;
                        break;
                    }
                }
                done = true;
            }
        }
    }

    // Add expression 'eq' with condition 'cond' to eq-group with 'baseName'
    // unless alread added.  The corresponding index in '_indices' will be
    // incremented if a new group is created.
    // 'eq_deps': pre-computed dependencies between equations.
    // Returns whether a new group was created.
    bool EqGroups::addExprToGroup(EqualsExprPtr eq,
                                  BoolExprPtr cond,
                                  const string& baseName,
                                  EqDepMap& eq_deps)
    {
        // Equation already added?
        if (_eqs_in_groups.count(eq))
            return false;

        // Look for existing group matching base-name and condition.
        EqGroup* target = 0;
        for (auto& eg : *this) {

            if (eg.baseName == baseName &&
                areExprsSame(eg.cond, cond)) {

                // Look for any dependencies that would prevent adding
                // eq to eg.
                bool is_dep = false;
                for (auto& eq2 : eg.getEqs()) {

                    for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                        if (eq_deps[dt].is_dep(eq, eq2)) {
#if DEBUG_ADD_EXPRS
                            cout << "addExprsFromGrid: not adding equation " <<
                                eq->makeQuotedStr() << " to " << eg.getDescription() <<
                                " because of dependency w/equation " <<
                                eq2->makeQuotedStr() << endl;
#endif
                            is_dep = true;
                            break;
                        }
                    }
                    if (is_dep)
                        break;
                }

                // Remember target group if found and no deps.
                if (!is_dep) {
                    target = &eg;
                    break;
                }
            }
        }
        
        // Make new group if needed.
        bool newGroup = false;
        if (!target) {
            EqGroup ne(*_dims);
            push_back(ne);
            target = &back();
            target->baseName = baseName;
            target->index = _indices[baseName]++;
            target->cond = cond;
            newGroup = true;
        
#if DEBUG_ADD_EXPRS
            cout << "Creating new " << target->getDescription() << endl;
#endif
        }

        // Add eq to target eq-group.
        assert(target);
#if DEBUG_ADD_EXPRS
        cout << "Adding " << eq->makeQuotedStr() <<
            " to " << target->getDescription() << endl;
#endif
        target->addEq(eq);
    
        // Remember eq and updated grid.
        _eqs_in_groups.insert(eq);
        _outGrids.insert(eq->getGrid());

        return newGroup;
    }

    // Divide all equations into eqGroups.
    // 'targets': string provided by user to specify grouping.
    // 'eq_deps': pre-computed dependencies between equations.
    void EqGroups::makeEqGroups(Eqs& allEqs,
                                const string& targets,
                                EqDepMap& eq_deps)
    {
        //auto& stepDim = _dims->_stepDim;
    
        // Handle each key-value pair in 'targets' string.
        ArgParser ap;
        ap.parseKeyValuePairs
            (targets, [&](const string& key, const string& value) {

                // Search allEqs for matches to current value.
                for (auto eq : allEqs.getEqs()) {

                    // Get name of updated grid.
                    auto gp = eq->getGrid();
                    assert(gp);
                    string gname = gp->getName();

                    // Does value appear in the grid name?
                    size_t np = gname.find(value);
                    if (np != string::npos) {

                        // Add equation.
                        addExprToGroup(eq, allEqs.getCond(eq), key, eq_deps);
                    }
                }
            });

        // Add all remaining equations.
        for (auto eq : allEqs.getEqs()) {

            // Add equation.
            addExprToGroup(eq, allEqs.getCond(eq), _basename_default, eq_deps);
        }

        // Find dependencies between eq-groups based on deps between their eqs.
        for (auto& eg1 : *this) {
            cout << " Checking dependencies of " <<
                eg1.getDescription() << "...\n";
            cout << "  Updating the following grid(s) with " <<
                eg1.getNumEqs() << " equation(s):";
            for (auto* g : eg1.getOutputGrids())
                cout << " " << g->getName();
            cout << endl;

            // Check to see if eg1 depends on other eq-groups.
            for (auto& eg2 : *this) {

                // Don't check against self.
                if (eg1.getName() == eg2.getName())
                    continue;

                if (eg1.setDepOn(certain_dep, eq_deps, eg2))
                    cout << "  Is dependent on " << eg2.getDescription(false) << endl;
                else if (eg1.setDepOn(possible_dep, eq_deps, eg2))
                    cout << "  May be dependent on " << eg2.getDescription(false) << endl;
            }
        }

        // Resort them based on dependencies.
        sort();
    }

    // Print stats from eqGroups.
    void EqGroups::printStats(ostream& os, const string& msg) {
        CounterVisitor cv;
        for (auto& eq : *this) {
            CounterVisitor ecv;
            eq.visitEqs(&ecv);
            cv += ecv;
        }
        cv.printStats(os, msg);
    }

    // Apply optimizations according to the 'settings'.
    void EqGroups::optimizeEqGroups(CompilerSettings& settings,
                                    const string& descr,
                                    bool printSets,
                                    ostream& os) {
        // print stats.
        string edescr = "for " + descr + " equation-group(s)";
        printStats(os, edescr);
    
        // Make a list of optimizations to apply to eqGroups.
        vector<OptVisitor*> opts;

        // CSE.
        if (settings._doCse)
            opts.push_back(new CseVisitor);

        // Operator combination.
        if (settings._doComb) {
            opts.push_back(new CombineVisitor);

            // Do CSE again after combination.
            // TODO: do this only if the combination did something.
            if (settings._doCse)
                opts.push_back(new CseVisitor);
        }

        // Apply opts.
        for (auto optimizer : opts) {

            visitEqs(optimizer);
            int numChanges = optimizer->getNumChanges();
            string odescr = "after applying " + optimizer->getName() + " to " +
                descr + " equation-group(s)";

            // Get new stats.
            if (numChanges)
                printStats(os, odescr);
            else
                os << "No changes " << odescr << '.' << endl;
        }

        // Final stats per equation group.
        if (printSets && size() > 1) {
            os << "Stats per equation-group:\n";
            for (auto eg : *this)
                eg.printStats(os, "for " + eg.getDescription());
        }
    }

    // Find the dimensions to be used based on the grids in
    // the solution and the settings from the cmd-line or API.
    void Dimensions::setDims(Grids& grids,
                             CompilerSettings& settings,
                             int vlen,                  // SIMD len based on CPU arch.
                             bool is_folding_efficient, // heuristic based on CPU arch.
                             ostream& os)
    {
        _domainDims.clear();
        _stencilDims.clear();
        _scalar.clear();
        _fold.clear();
        _clusterPts.clear();
        _clusterMults.clear();
        _miscDims.clear();

        // Get dims from grids.
        for (auto gp : grids) {
                
            // Dimensions in this grid.
            for (auto dim : gp->getDims()) {
                auto& dname = dim->getName();
                auto type = dim->getType();

                switch (type) {

                case STEP_INDEX:
                    if (_stepDim.length() && _stepDim != dname) {
                        cerr << "Error: step dimensions '" << _stepDim <<
                            "' and '" << dname << "' found; only one allowed.\n";
                        exit(1);
                    }
                    _stepDim = dname;
                    _stencilDims.addDimBack(dname, 0);
                    break;

                case DOMAIN_INDEX:
                    _domainDims.addDimBack(dname, 0);
                    _stencilDims.addDimBack(dname, 0);
                    _scalar.addDimBack(dname, 1);
                    _fold.addDimBack(dname, 1);
                    _clusterMults.addDimBack(dname, 1);
                    break;

                case MISC_INDEX:
                    _miscDims.addDimBack(dname, 0);
                    break;

                default:
                    cerr << "Error: unexpected dim type " << type << ".\n";
                    exit(1);
                }
            }
        }
        if (_stepDim.length() == 0) {
            cerr << "Error: no step dimension defined.\n";
            exit(1);
        }
        if (!_domainDims.getNumDims()) {
            cerr << "Error: no domain dimensions defined.\n";
            exit(1);
        }
        
        // Layout of fold.
        _fold.setFirstInner(settings._firstInner);
        
        os << "Step dimension: " << _stepDim << endl;
        os << "Domain dimension(s): " << _domainDims.makeDimStr() << endl;
    
        // Set fold lengths based on cmd-line options.
        IntTuple foldGT1;    // fold dimensions > 1.
        for (auto& dim : settings._foldOptions.getDims()) {
            auto& dname = dim.getName();
            int sz = dim.getVal();

            // Nothing to do for fold < 2.
            if (sz <= 1)
                continue;

            // Domain dim?
            if (!_domainDims.lookup(dname)) {
                os << "Warning: fold in '" << dname <<
                    "' dim ignored because it is not a domain dim.\n";
                continue;
            }

            // Set size.
            _fold.addDimBack(dname, sz);
            foldGT1.addDimBack(dname, sz);
        }

        // Make sure folds cover vlen (unless vlen is 1).
        if (vlen > 1 && _fold.product() != vlen) {
            if (_fold.product() > 1)
                os << "Notice: adjusting requested fold to achieve SIMD length of " <<
                    vlen << ".\n";

            // Heuristics to determine which dims to modify.
            IntTuple targets = foldGT1; // start with specified ones >1.
            const int nTargets = is_folding_efficient ? 2 : 1; // desired num targets.
            int fdims = _fold.getNumDims();
            if (targets.getNumDims() < nTargets && fdims > 1)
                targets.addDimBack(_fold.getDim(fdims - 2)); // 2nd from last.
            if (targets.getNumDims() < nTargets && fdims > 2)
                targets.addDimBack(_fold.getDim(fdims - 3)); // 3rd from last.
            if (targets.getNumDims() < nTargets)
                targets = _fold; // all.

            // Heuristic: incrementally increase targets by powers of 2.
            _fold.setValsSame(1);
            for (int n = 1; _fold.product() < vlen; n++) {
                for (auto i : targets.getDims()) {
                    auto& dname = i.getName();
                    if (_fold.product() < vlen)
                        _fold.setVal(dname, 1 << n);
                }
            }

            // Still wrong?
            if (_fold.product() != vlen) {
                _fold.setValsSame(1);

                // Heuristic: set first target to vlen.
                if (targets.getNumDims()) {
                    auto& dname = targets.getDim(0).getName();
                    _fold.setVal(dname, vlen);
                }
            }

            // Still wrong?
            if (_fold.product() != vlen) {
                _fold.setValsSame(1);
                os << "Warning: not able to adjust fold.\n";
            }

            // Fix foldGT1.
            foldGT1.clear();
            for (auto i : _fold.getDims()) {
                auto& dname = i.getName();
                auto& val = i.getVal();
                if (val > 1)
                    foldGT1.addDimBack(dname, val);
            }
        }
        os << " Number of SIMD elements: " << vlen << endl;
        os << " Vector-fold dimension(s) and point-size(s): " <<
            _fold.makeDimValStr(" * ") << endl;

        // Checks for unaligned loads.
        if (settings._allowUnalignedLoads) {
            if (foldGT1.size() > 1) {
                cerr << "Error: attempt to allow unaligned loads when there are " <<
                    foldGT1.size() << " dimensions in the vector-fold that are > 1." << endl;
                exit(1);
            }
            else if (foldGT1.size() > 0)
                cerr << "Notice: memory layout MUST have unit-stride in " <<
                    foldGT1.makeDimStr() << " dimension!" << endl;
        }

        // Create final cluster lengths based on cmd-line options.
        for (auto& dim : settings._clusterOptions.getDims()) {
            auto& dname = dim.getName();
            int mult = dim.getVal();

            // Nothing to do for mult < 2.
            if (mult <= 1)
                continue;

            // Does it exist anywhere?
            if (!_domainDims.lookup(dname)) {
                os << "Warning: cluster-multiplier in '" << dname <<
                    "' dim ignored because it's not a domain dim.\n";
                continue;
            }

            // Set the size.
            _clusterMults.addDimBack(dname, mult);
        }
        _clusterPts = _fold.multElements(_clusterMults);
    
        os << " Cluster dimension(s) and multiplier(s): " <<
            _clusterMults.makeDimValStr(" * ") << endl;
        os << " Cluster dimension(s) and point-size(s): " <<
            _clusterPts.makeDimValStr(" * ") << endl;
        if (_miscDims.getNumDims())
            os << "Misc dimension(s): " << _miscDims.makeDimStr() << endl;
    }
} // namespace yask.
